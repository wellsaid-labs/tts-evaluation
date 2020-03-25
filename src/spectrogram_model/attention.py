import math

import torch

from hparams import configurable
from hparams import HParam
from torch import nn
from torchnlp.nn import LockedDropout


class LocationSensitiveAttention(nn.Module):
    """ Query using the Bahdanau attention mechanism with additional location features.

    SOURCE (Tacotron 2):
        The encoder output is consumed by an attention network which summarizes the full encoded
        sequence as a fixed-length context vector for each decoder output step. We use the
        location-sensitive attention from [21], which extends the additive attention mechanism
        [22] to use cumulative attention weights from previous decoder time steps as an additional
        feature. This encourages the model to move forward consistently through the input,
        mitigating potential failure modes where some subsequences are repeated or ignored by the
        decoder. Attention probabilities are computed after projecting inputs and location
        features to 128-dimensional hidden representations. Location features are computed using
        32 1-D convolution filters of length 31.

    SOURCE (Attention-Based Models for Speech Recognition):
        This is achieved by adding as inputs to the attention mechanism auxiliary convolutional
        features which are extracted by convolving the attention weights from the previous step
        with trainable filters.

        ...

        We extend this content-based attention mechanism of the original model to be location-aware
        by making it take into account the alignment produced at the previous step. First, we
        extract k vectors fi,j ∈ R^k for every position j of the previous alignment αi−1 by
        convolving it with a matrix F ∈ R^k×r.

    Reference:
        * Tacotron 2 Paper:
          https://arxiv.org/pdf/1712.05884.pdf
        * Attention-Based Models for Speech Recognition
          https://arxiv.org/pdf/1506.07503.pdf

    Args:
        query_hidden_size (int): The hidden size of the query input.
        hidden_size (int): The hidden size of the hidden attention features.
        convolution_filter_size (int): Size of the convolving kernel applied to the cumulative
            alignment.
        dropout (float): The dropout probability.
        initializer_range (float): The normal initialization standard deviation.
    """

    @configurable
    def __init__(self,
                 query_hidden_size,
                 hidden_size=HParam(),
                 convolution_filter_size=HParam(),
                 dropout=HParam(),
                 initializer_range=HParam()):
        super().__init__()

        # LEARN MORE:
        # https://datascience.stackexchange.com/questions/23183/why-convolutions-always-use-odd-numbers-as-filter-size
        assert convolution_filter_size % 2 == 1, '`convolution_filter_size` must be odd'

        self.dropout = dropout
        self.hidden_size = hidden_size
        self.initializer_range = initializer_range

        self.alignment_conv_padding = int((convolution_filter_size - 1) / 2)
        self.alignment_conv = nn.Conv1d(
            in_channels=1, out_channels=hidden_size, kernel_size=convolution_filter_size, padding=0)

        self.project_query = nn.Linear(query_hidden_size, hidden_size)
        self.project_scores = nn.Sequential(
            LockedDropout(dropout), nn.Linear(hidden_size, 1, bias=False))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.project_scores[1].weight, std=self.initializer_range)
        nn.init.normal_(self.project_query.weight, std=self.initializer_range)
        nn.init.normal_(self.alignment_conv.weight, std=self.initializer_range)
        nn.init.zeros_(self.project_query.bias)
        nn.init.zeros_(self.alignment_conv.bias)

    def forward(self,
                encoded_tokens,
                tokens_mask,
                query,
                cumulative_alignment=None,
                initial_cumulative_alignment=None):
        """
        Args:
            encoded_tokens (torch.FloatTensor [num_tokens, batch_size, hidden_size]): Batched set of
                encoded sequences.
            tokens_mask (torch.BoolTensor [batch_size, num_tokens]): Binary mask where zero's
                represent padding in ``encoded_tokens``.
            query (torch.FloatTensor [1, batch_size, query_hidden_size]): Query vector used to score
                individual token vectors.
            cumulative_alignment (torch.FloatTensor [batch_size, num_tokens], optional): Cumlative
                attention alignment from the last queries. If this vector is not included, the
                ``cumulative_alignment`` defaults to a zero vector.
            initial_cumulative_alignment (torch.FloatTensor [batch_size, 1]): The left-side
                padding value for the `alignment_conv`. This can also be interpreted as the
                cumulative alignment for the former tokens.

        Returns:
            context (torch.FloatTensor [batch_size, hidden_size]): Computed attention
                context vector.
            cumulative_alignment (torch.FloatTensor [batch_size, num_tokens]): Cumlative attention
                alignment vector.
            alignment (torch.FloatTensor [batch_size, num_tokens]): Computed attention alignment
                vector.
        """
        num_tokens, batch_size, _ = encoded_tokens.shape
        device = encoded_tokens.device

        # NOTE: Attention alignment is sometimes refered to as attention weights.
        if cumulative_alignment is None:
            cumulative_alignment = torch.zeros(batch_size, num_tokens, device=device)
        if initial_cumulative_alignment is None:
            initial_cumulative_alignment = torch.zeros(batch_size, 1, device=device)

        cumulative_alignment = cumulative_alignment.masked_fill(~tokens_mask, 0)

        # [batch_size, num_tokens] → [batch_size, 1, num_tokens]
        location_features = cumulative_alignment.unsqueeze(1).detach()

        # NOTE: Add `self.alignment_conv_padding` to both sides.
        initial_cumulative_alignment = initial_cumulative_alignment.unsqueeze(-1).expand(
            -1, -1, self.alignment_conv_padding)
        location_features = torch.cat([initial_cumulative_alignment, location_features], dim=-1)
        location_features = torch.nn.functional.pad(
            location_features, (0, self.alignment_conv_padding), mode='constant', value=0.0)

        # [batch_size, 1, num_tokens] → [batch_size, hidden_size, num_tokens]
        location_features = self.alignment_conv(location_features)

        # [1, batch_size, query_hidden_size] → [batch_size, hidden_size, 1]
        query = self.project_query(query).view(batch_size, self.hidden_size, 1)

        # [batch_size, hidden_size, num_tokens]
        location_features = torch.tanh(location_features + query)

        # [batch_size, hidden_size, num_tokens] →
        # [num_tokens, batch_size, hidden_size] →
        # [num_tokens, batch_size]
        score = self.project_scores(location_features.permute(2, 0, 1)).squeeze(2).transpose(0, 1)

        score.data.masked_fill_(~tokens_mask, -math.inf)

        # [batch_size, num_tokens] → [batch_size, num_tokens]
        alignment = torch.softmax(score, dim=1)

        # NOTE: Transpose and unsqueeze to fit the requirements for ``torch.bmm``
        # [num_tokens, batch_size, hidden_size] →
        # [batch_size, num_tokens, hidden_size]
        encoded_tokens = encoded_tokens.transpose(0, 1)

        # alignment [batch_size (b), 1 (n), num_tokens (m)] (bmm)
        # encoded_tokens [batch_size (b), num_tokens (m), hidden_size (p)] →
        # [batch_size (b), 1 (n), hidden_size (p)]
        context = torch.bmm(alignment.unsqueeze(1), encoded_tokens)

        # [batch_size, 1, hidden_size] → [batch_size, hidden_size]
        context = context.squeeze(1)

        # [batch_size, num_tokens] + [batch_size, num_tokens] → [batch_size, num_tokens]
        cumulative_alignment = cumulative_alignment + alignment

        return context, cumulative_alignment, alignment

import math

import torch

from hparams import configurable
from hparams import HParam
from torch import nn


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
        query_hidden_size (int): The hidden size of the query to expect.
        hidden_size (int): The hidden size of the attention module dictating context, query,
            and location features size.
        num_convolution_filters (odd :clas:`int`): Number of dimensions (channels)
            produced by the convolution.
        convolution_filter_size (int): Size of the convolving kernel.
    """

    @configurable
    def __init__(self,
                 query_hidden_size,
                 hidden_size,
                 num_convolution_filters=HParam(),
                 convolution_filter_size=HParam()):

        super().__init__()
        # LEARN MORE:
        # https://datascience.stackexchange.com/questions/23183/why-convolutions-always-use-odd-numbers-as-filter-size
        assert convolution_filter_size % 2 == 1, '`convolution_filter_size` must be odd'

        self.alignment_conv_padding = int((convolution_filter_size - 1) / 2)
        self.alignment_conv = nn.Conv1d(
            in_channels=1,
            out_channels=num_convolution_filters + 1,
            kernel_size=convolution_filter_size,
            padding=0)
        self.project_query = nn.Sequential(
            nn.Linear(query_hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size))
        self.project_alignment = nn.Linear(num_convolution_filters, hidden_size, bias=False)
        self.project_scores = nn.Linear(hidden_size, 1, bias=False)
        self.softmax = nn.Softmax(dim=1)

        # Initialize Weights
        nn.init.xavier_uniform_(self.project_scores.weight, gain=nn.init.calculate_gain('linear'))

    def score(self, encoded_tokens, tokens_mask, query, location_features):
        """ Compute addative attention score with location features.

        Args:
            encoded_tokens (torch.FloatTensor [num_tokens, batch_size, hidden_size]):
                Batched set of encoded sequences.
            tokens_mask (torch.BoolTensor [batch_size, num_tokens]): Binary mask where zeros's
                represent padding in ``encoded_tokens``.
            query (torch.FloatTensor [1, batch_size, hidden_size]): Query vector used to score
                individual token vectors.
            location_features (torch.FloatTensor [num_tokens, batch_size, hidden_size]): Location
                features extracted from the cumulative attention alignment from the last queries.

        Returns:
            alignment (torch.FloatTensor [batch_size, num_tokens]): Alignment over
                ``encoded_tokens``
        """
        num_tokens, batch_size, hidden_size = encoded_tokens.shape

        # [1, batch_size, hidden_size] → [num_tokens, batch_size, hidden_size]
        query = query.expand(num_tokens, batch_size, hidden_size)

        # score [num_tokens, batch_size, hidden_size]
        # ei,j = w * tanh(W * si−1 + V * hj + U * fi,j + b)
        score = (query + location_features).tanh_()

        del location_features  # Clear memory
        del query  # Clear memory

        # [num_tokens, batch_size, hidden_size] → [batch_size, num_tokens, hidden_size]
        score = score.transpose(0, 1)

        # [batch_size, num_tokens, hidden_size] → [batch_size, num_tokens]
        score = self.project_scores(score).squeeze(2)

        # Mask encoded tokens padding
        score.data.masked_fill_(~tokens_mask, -math.inf)

        # [batch_size, num_tokens] → [batch_size, num_tokens]
        alignment = self.softmax(score)

        del score  # Clear memory

        return alignment

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
        cumulative_alignment = torch.zeros(
            batch_size, num_tokens, dtype=torch.float,
            device=device) if cumulative_alignment is None else cumulative_alignment
        initial_cumulative_alignment = torch.zeros(
            batch_size, 1,
            device=device) if initial_cumulative_alignment is None else initial_cumulative_alignment

        cumulative_alignment = cumulative_alignment.masked_fill(~tokens_mask, 0)

        # [batch_size, num_tokens] → [batch_size, 1, num_tokens]
        location_features = cumulative_alignment.unsqueeze(1)

        # Add `self.alignment_conv_padding` to both sides.
        initial_cumulative_alignment = initial_cumulative_alignment.unsqueeze(-1).expand(
            -1, -1, self.alignment_conv_padding)
        location_features = torch.cat([initial_cumulative_alignment, location_features], dim=-1)
        location_features = torch.nn.functional.pad(
            location_features, (0, self.alignment_conv_padding), mode='constant', value=0.0)

        # [batch_size, 1, num_tokens] → [batch_size, num_convolution_filters + 1, num_tokens]
        location_features, location_gate = self.alignment_conv(location_features).split(
            [self.alignment_conv.out_channels - 1, 1], dim=1)
        # [batch_size, num_convolution_filters, num_tokens] →
        # [num_tokens, batch_size, num_convolution_filters]
        location_features = location_features.permute(2, 0, 1)
        # [num_tokens, batch_size, num_convolution_filters] →
        # [num_tokens, batch_size, hidden_size]
        location_features = self.project_alignment(location_features)

        # [1, batch_size, query_hidden_size] → [1, batch_size, hidden_size]
        query = self.project_query(query)

        # alignment [batch_size, num_tokens]
        alignment = self.score(encoded_tokens, tokens_mask, query, location_features)

        # Transpose and unsqueeze to fit the requirements for ``torch.bmm``
        # [num_tokens, batch_size, hidden_size] →
        # [batch_size, num_tokens, hidden_size]
        encoded_tokens = encoded_tokens.transpose(0, 1)

        # alignment [batch_size (b), 1 (n), num_tokens (m)] (bmm)
        # encoded_tokens [batch_size (b), num_tokens (m), hidden_size (p)] →
        # [batch_size (b), 1 (n), hidden_size (p)]
        context = torch.bmm(alignment.unsqueeze(1), encoded_tokens)

        # Squeeze extra single dimension
        # [batch_size, 1, hidden_size] → [batch_size, hidden_size]
        context = context.squeeze(1)

        # TODO: Does there exist a more effective gating mechanism that'd be more precise? Since
        # `self.alignment_conv` computes the gate it'd loosely be based on the ratio of zeros to
        # nonzeros.
        # TODO: At the moment, we wouldn't see the affects of this gate because we don't
        # visualize the gated alignment; therefore, in order to improve it further that'd
        # likely be beneficial.
        # [batch_size, num_tokens] + [batch_size, num_tokens] → [batch_size, num_tokens]
        cumulative_alignment = cumulative_alignment + alignment * torch.sigmoid(
            location_gate).squeeze(1)

        # Normalize energies to weights in range 0 to 1, resize to 1 x 1 x seq_len
        return context, cumulative_alignment, alignment

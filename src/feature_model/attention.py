import torch

from torch import nn
from torch.nn import functional

from src.utils.configurable import configurable


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
        encoder_hidden_size (int): Hidden size of the encoder used; for reference.
        query_hidden_size (int): The hidden size of the query to expect.
        alignment_hidden_size (int): The hidden size of the alignment to project to.
        num_convolution_filters (odd :clas:`int`, optional): Number of dimensions (channels)
            produced by the convolution.
        convolution_filter_size (int, optional): Size of the convolving kernel.
    """

    @configurable
    def __init__(self,
                 encoder_hidden_size=512,
                 query_hidden_size=1024,
                 hidden_size=128,
                 num_convolution_filters=32,
                 convolution_filter_size=31):

        super().__init__()
        # LEARN MORE:
        # https://datascience.stackexchange.com/questions/23183/why-convolutions-always-use-odd-numbers-as-filter-size
        assert convolution_filter_size % 2 == 1, ('`convolution_filter_size` must be odd')

        self.alignment_conv = nn.Conv1d(
            in_channels=1,
            out_channels=num_convolution_filters,
            kernel_size=convolution_filter_size,
            padding=int((convolution_filter_size - 1) / 2))
        self.project_query = nn.Linear(query_hidden_size, hidden_size)
        self.project_alignment = nn.Linear(num_convolution_filters, hidden_size)
        self.score_weights = nn.Parameter(torch.FloatTensor(1, hidden_size, 1))
        self.score_bias = nn.Parameter(torch.FloatTensor(1, 1, hidden_size).zero_())
        self.softmax = nn.Softmax(dim=1)

        # Initialize Weights
        nn.init.xavier_uniform_(self.score_weights, gain=nn.init.calculate_gain('linear'))

    def score(self, encoded_tokens, tokens_mask, query, location_features):
        """ Compute addative attention score with location features.

        Args:
            encoded_tokens (torch.FloatTensor [num_tokens, batch_size, hidden_size]):
                Batched set of encoded sequences.
            tokens_mask (torch.FloatTensor [batch_size, num_tokens]): Binary mask where one's
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

        # [1, 1, hidden_size] → [num_tokens, batch_size, hidden_size]
        score_bias = self.score_bias.expand(num_tokens, batch_size, hidden_size)

        # score [num_tokens, batch_size, hidden_size]
        # ei,j = w * tanh(W * si−1 + V * hj + U * fi,j + b)
        score = functional.tanh(encoded_tokens + query + location_features + score_bias)

        del location_features  # Clear memory
        del query  # Clear memory

        # [1, hidden_size, 1] →
        # [batch_size, hidden_size, 1]
        score_weights = self.score_weights.expand(batch_size, hidden_size, 1)

        # [num_tokens, batch_size, hidden_size] → [batch_size, num_tokens, hidden_size]
        score = score.transpose(0, 1)

        # [batch_size (b), num_tokens (n), hidden_size (m)] (bmm)
        # [batch_size (b), hidden_size (m), 1 (p)] →
        # [batch_size (b), num_tokens (n), 1 (p)]
        score = torch.bmm(score, score_weights)

        del score_weights  # Clear memory

        # Squeeze extra single dimension
        # [batch_size, num_tokens, 1] → [batch_size, num_tokens]
        score = score.squeeze(2)

        # Mask encoded tokens padding
        score.data.masked_fill_(tokens_mask, float('-inf'))

        # [batch_size, num_tokens] → [batch_size, num_tokens]
        alignment = self.softmax(score)

        del score  # Clear memory

        return alignment

    def forward(self, encoded_tokens, tokens_mask, query, cumulative_alignment=None):
        """
        Args:
            encoded_tokens (torch.FloatTensor [num_tokens, batch_size, attention_hidden_size]):
                Batched set of encoded sequences.
            tokens_mask (torch.FloatTensor [batch_size, num_tokens]): Binary mask where one's
                represent padding in ``encoded_tokens``.
            query (torch.FloatTensor [1, batch_size, query_hidden_size]): Query vector used to score
                individual token vectors.
            cumulative_alignment (torch.FloatTensor [batch_size, num_tokens], optional): Cumlative
                attention alignment from the last queries. If this vector is not included, the
                ``cumulative_alignment`` defaults to a zero vector.

        Returns:
            context (torch.FloatTensor [batch_size, hidden_size]): Computed attention
                context vector.
            cumulative_alignment (torch.FloatTensor [batch_size, num_tokens]): Cumlative attention
                alignment vector.
            alignment (torch.FloatTensor [batch_size, num_tokens]): Computed attention alignment
                vector.
        """
        num_tokens, batch_size, _ = encoded_tokens.shape

        if cumulative_alignment is None:
            # Attention alignment, sometimes refered to as attention weights.
            tensor = torch.cuda.FloatTensor if encoded_tokens.is_cuda else torch.FloatTensor
            cumulative_alignment = tensor(batch_size, num_tokens).zero_()

        # [batch_size, num_tokens] → [batch_size, 1, num_tokens]
        location_features = cumulative_alignment.unsqueeze(1)
        # [batch_size, 1, num_tokens] → [batch_size, num_convolution_filters, num_tokens]
        location_features = self.alignment_conv(location_features)
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
        # [batch_size, 1, encoder_hidden_size] → [batch_size, encoder_hidden_size]
        context = context.squeeze(1)

        # [batch_size, num_tokens] + [batch_size, num_tokens] → [batch_size, num_tokens]
        cumulative_alignment = cumulative_alignment + alignment

        # Normalize energies to weights in range 0 to 1, resize to 1 x 1 x seq_len
        return context, cumulative_alignment, alignment

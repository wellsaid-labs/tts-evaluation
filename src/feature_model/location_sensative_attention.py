from functools import partial

from torch import nn

from src.configurable import configurable


class LocationSensitiveAttention(nn.Module):
    """ Query using the attention mechanism.

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
        We extend this content-based attention mechanism of the original model to be location-aware
        by making it take into account the alignment produced at the previous step. First, we
        extract k vectors fi,j ∈ R^k for every position j of the previous alignment αi−1 by
        convolving it with a matrix F ∈ R^k×r.
    """

    @configurable
    def __init__(self,
                 encoder_hidden_size=512,
                 num_convolution_filters=32,
                 convolution_filter_size=31):
        # LEARN MORE:
        # https://datascience.stackexchange.com/questions/23183/why-convolutions-always-use-odd-numbers-as-filter-size
        assert convolution_filter_size % 2 == 1, ('`convolution_filter_size` must be odd')

        # self.conv = nn.Conv1d(
        #     in_channels=token_dim,
        #     out_channels=num_convolution_filters,
        #     kernel_size=convolution_filter_size,
        #     padding=int((convolution_filter_size - 1) / 2))

        # self.linear_in = nn.Linear(dimensions, dimensions, bias=False)
        pass

    def forward(self, encoded_tokens, query, previous_alignment=None):
        """
        Args:
            encoded_tokens (torch.FloatTensor [num_tokens, batch_size, hidden_size]): Batched set of
                encoded sequences.
            query (torch.FloatTensor [batch_size, hidden_size]): Query vector used to score
                individual token vectors.
            previous_alignment (torch.FloatTensor [num_tokens, batch_size]): Attention alignment
                from the last query.

        Returns:
            context (torch.FloatTensor [batch_size, hidden_size])
            alignment (torch.FloatTensor [num_tokens, batch_size])
        """
        if previous_alignment is None:
            # Attention alignment, sometimes refered to as attention weights.
            previous_alignment = torch.autograd.Variable(
                torch.LongTensor(encoded_tokens.shape[0], encoded_tokens.shape[1]).zero_(),
                requires_grad=False)
        # query = query.view(batch_size * output_len, dimensions)
        # query = self.linear_in(query)
        # query = query.view(batch_size, output_len, dimensions)
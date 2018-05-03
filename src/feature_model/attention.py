import torch

from torch import nn
from torch.autograd import Variable

from src.configurable import configurable


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

    # TODO: Add attention visualization for debugging

    @configurable
    def __init__(self,
                 encoder_hidden_size=512,
                 query_hidden_size=128,
                 alignment_hidden_size=128,
                 num_convolution_filters=32,
                 convolution_filter_size=31):

        super(LocationSensitiveAttention, self).__init__()
        # LEARN MORE:
        # https://datascience.stackexchange.com/questions/23183/why-convolutions-always-use-odd-numbers-as-filter-size
        assert convolution_filter_size % 2 == 1, ('`convolution_filter_size` must be odd')

        self.alignment_conv = nn.Conv1d(
            in_channels=1,
            out_channels=num_convolution_filters,
            kernel_size=convolution_filter_size,
            padding=int((convolution_filter_size - 1) / 2))
        self.project_alignment = nn.Linear(
            in_features=num_convolution_filters, out_features=alignment_hidden_size)

        self.score_linear = nn.Sequential(
            nn.Linear(query_hidden_size + encoder_hidden_size + alignment_hidden_size,
                      encoder_hidden_size), nn.Tanh())
        self.score_parameter = nn.Parameter(torch.FloatTensor(1, encoder_hidden_size, 1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, encoded_tokens, query, last_alignment=None):
        """
        Args:
            encoded_tokens (torch.FloatTensor [num_tokens, batch_size, encoder_hidden_size]):
                Batched set of encoded sequences.
            query (torch.FloatTensor [batch_size, query_hidden_size]): Query vector used to score
                individual token vectors.
            last_alignment (torch.FloatTensor [batch_size, num_tokens], optional): Attention
                alignment from the last query. If this vector is not included, the
                ``last_alignment`` defaults to a zero vector.

        Returns:
            context (torch.FloatTensor [batch_size, encoder_hidden_size]): Computed attention
                context vector.
            alignment (torch.FloatTensor [batch_size, num_tokens]): Computed attention alignment
                vector.
        """
        if last_alignment is None:
            # Attention alignment, sometimes refered to as attention weights.
            num_tokens, batch_size, _ = encoded_tokens.shape
            last_alignment = Variable(
                torch.FloatTensor(batch_size, num_tokens).zero_(), requires_grad=False)

        # [batch_size, num_tokens] → [batch_size, 1, num_tokens]
        last_alignment = last_alignment.unsqueeze(1)
        # [batch_size, 1, num_tokens] → [batch_size, num_convolution_filters, num_tokens]
        last_alignment = self.alignment_conv(last_alignment)
        # [batch_size, num_convolution_filters, num_tokens] →
        # [num_tokens, batch_size, num_convolution_filters]
        last_alignment = last_alignment.permute(2, 0, 1)
        # [num_tokens, batch_size, num_convolution_filters] →
        # [num_tokens, batch_size, alignment_hidden_size]
        last_alignment = self.project_alignment(last_alignment)

        # [batch_size, query_hidden_size] → [1, batch_size, query_hidden_size]
        query = query.unsqueeze(0)
        num_tokens, batch_size, _ = last_alignment.shape
        # [1, batch_size, query_hidden_size] → [num_tokens, batch_size, query_hidden_size]
        query = query.expand(num_tokens, -1, -1)

        # encoded_tokens [num_tokens, batch_size, encoder_hidden_size] (concat)
        # query [num_tokens, batch_size, query_hidden_size] (concat)
        # last_alignment [num_tokens, batch_size, alignment_hidden_size] →
        # [num_tokens, batch_size, alignment_hidden_size + query_hidden_size + encoder_hidden_size]
        concat = torch.cat((encoded_tokens, query, last_alignment), -1)
        # [num_tokens, batch_size, alignment_hidden_size + query_hidden_size + encoder_hidden_size]
        # → [num_tokens, batch_size, encoder_hidden_size]
        score = self.score_linear(concat)

        # Transpose and expand to fit the requirements for ``torch.bmm``
        # [num_tokens, batch_size, encoder_hidden_size] →
        # [batch_size, num_tokens, encoder_hidden_size]
        score = score.transpose(0, 1)
        # [1, encoder_hidden_size, 1] →
        # [batch_size, encoder_hidden_size, 1]
        score_parameter = self.score_parameter.expand(batch_size, -1, -1)

        # [batch_size (b), num_tokens (n), encoder_hidden_size (m)] (bmm)
        # [batch_size (b), encoder_hidden_size (m), 1 (p)] →
        # [batch_size (b), num_tokens (n), 1 (p)]
        score = torch.bmm(score, score_parameter)

        # Squeeze extra single dimension
        # [batch_size, num_tokens, 1] → [batch_size, num_tokens]
        score = score.squeeze(2)

        # [batch_size, num_tokens] → [batch_size, num_tokens]
        alignment = self.softmax(score)

        # Transpose and unsqueeze to fit the requirements for ``torch.bmm``
        # [num_tokens, batch_size, encoder_hidden_size] →
        # [batch_size, num_tokens, encoder_hidden_size]
        encoded_tokens = encoded_tokens.transpose(0, 1)
        # [batch_size, num_tokens] → [batch_size, 1, num_tokens]
        alignment = alignment.unsqueeze(1)

        # alignment [batch_size (b), 1 (n), num_tokens (m)] (bmm)
        # encoded_tokens [batch_size (b), num_tokens (m), encoder_hidden_size (p)] →
        # [batch_size (b), 1 (n), encoder_hidden_size (p)]
        context = torch.bmm(alignment, encoded_tokens)

        # Squeeze extra single dimension
        # [batch_size, 1, encoder_hidden_size] → [batch_size, encoder_hidden_size]
        context = context.squeeze(1)
        # Squeeze extra single dimension
        # [batch_size, 1, num_tokens] → [batch_size, num_tokens]
        alignment = alignment.squeeze(1)

        # Normalize energies to weights in range 0 to 1, resize to 1 x 1 x seq_len
        return context, alignment

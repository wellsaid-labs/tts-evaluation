import torch

from torch import nn


class StrippedGRU(nn.Module):
    """ Stripped GRU is a GRU that does not apply a linear layer to the input.

    Args:
        size (int): Hidden size of the GRU.
    """

    def __init__(self, size):
        super(StrippedGRU, self).__init__()

        self.size = size
        # NOTE: If ``input_size`` is equal to ``size`` then this GPU check fails:
        #   File "/home/michaelp/.local/lib/python3.5/site-packages/torch/nn/modules/rnn.py", line
        #   105, in flatten_parameters
        #       self.batch_first, bool(self.bidirectional))
        #   RuntimeError: invalid argument 2: size '[2408448 x 1]' is invalid for input with 7225344
        #   elements at /pytorch/aten/src/TH/THStorage.c:41
        # and this CPU check:
        #   input.size(-1) must be equal to input_size.
        self.gru = nn.GRU(input_size=size * 3, hidden_size=size, num_layers=1)

        # HACK: ``gru.weight_ih_l0`` size for a GRU should be [input_size * 3, hidden_size] but
        # we are able to get away with an identity matrix here instead
        self.gru.weight_ih_l0 = torch.nn.Parameter(
            torch.eye(size * 3, size * 3), requires_grad=False)

    def forward(self, input_, hidden=None):
        assert input_.shape[2] == self.size * 3
        assert hidden is None or (hidden.shape[0] == 1 and hidden.shape[2] == self.size)
        return self.gru(input_, hidden)

import torch

from torch import nn
from torch.nn.functional import sigmoid
from torch.nn.functional import tanh
from tqdm import tqdm

from src.signal_model.upsample import ConditionalFeaturesUpsample
from src.signal_model.stripped_gru import StrippedGRU
from src.utils.configurable import configurable
from src.utils import split_signal


class WaveRNN(nn.Module):
    """ WaveRNN as defined in "Efficient Neural Audio Synthesis".

    References:
        * Efficient Neural Audio Synthesis
          https://arxiv.org/pdf/1802.08435.pdf

    TODO:
        * Add ONNX support for inference. To do so we'd need to substitue ``torch.repeat`` and
          ``torch.stack`` because they are not supported.

    Args:
        hidden_size (int): GRU hidden state size.
        bits (int): Number of bits  Number of categories to predict for coarse and fine variables.
        upsample_convs (list of int): Size of convolution layers used to upsample local features
            (e.g. 256 frames x 4 x ...).
        upsample_repeat (int): Number of times to repeat frames, another upsampling technique.
        local_features_size (int): Dimensionality of local features.
    """

    @configurable
    def __init__(self,
                 hidden_size=896,
                 bits=16,
                 upsample_convs=[4],
                 upsample_repeat=75,
                 local_features_size=80):
        super(WaveRNN, self).__init__()

        assert hidden_size % 2 == 0, "Hidden size must be even."
        assert bits % 2 == 0, "Bits must be even for a double softmax"
        self.bins = int(2**(bits / 2))  # Encode ``bits`` with double softmax of ``bits / 2``
        self.size = hidden_size
        self.half_size = int(self.size / 2)

        relu_gain = torch.nn.init.calculate_gain('relu')
        # Output fully connected layers
        self.to_bins_coarse = nn.Sequential(
            nn.Linear(self.half_size, self.half_size), nn.ReLU(),
            nn.Linear(self.half_size, self.bins))
        torch.nn.init.orthogonal_(self.to_bins_coarse[0].weight, gain=relu_gain)

        self.to_bins_fine = nn.Sequential(
            nn.Linear(self.half_size, self.half_size), nn.ReLU(),
            nn.Linear(self.half_size, self.bins))
        # NOTE: Orthogonal is inspired by:
        # https://github.com/soroushmehr/sampleRNN_ICLR2017/blob/master/lib/ops.py#L75
        torch.nn.init.orthogonal_(self.to_bins_fine[0].weight, gain=relu_gain)

        # Input fully connected layers
        self.project_coarse_input = nn.Linear(2, 3 * self.half_size, bias=False)
        self.project_fine_input = nn.Linear(3, 3 * self.half_size, bias=False)

        self.conditional_features_upsample = ConditionalFeaturesUpsample(
            in_channels=local_features_size,
            out_channels=self.size,
            upsample_repeat=upsample_repeat,
            upsample_convs=upsample_convs,
            num_layers=3,
            upsample_chunks=1)

        self.stripped_gru = StrippedGRU(self.size)

    def _scale(self, tensor):
        """ Scale from [0, self.bins - 1] to [-1.0, 1.0].

        Args:
            tensor (torch.FloatTensor): Tensor between the range of [0, self.bins]

        Returns:
            tensor (torch.FloatTensor): Tensor between the range of [-1.0, 1.0]
        """
        assert torch.min(tensor) >= 0 and torch.max(tensor) < self.bins
        return tensor.float() / ((self.bins - 1) / 2) - 1.0

    def forward(self, local_features, input_signal=None, target_coarse=None):
        """
        Args:
            local_features (torch.FloatTensor [batch_size, local_length, local_features_size]):
                Local feature to condition signal generation (e.g. spectrogram).
            input_signal (torch.FloatTensor [batch_size, signal_length, 2], optional): Course
                ``signal[:, :, 0]`` and fines values ``signal[:, :, 1]`` used for teacher forcing
                each between the range [-1, 1].
            target_coarse (torch.FloatTensor [batch_size, signal_length, 1], optional): Same as the
                input signal but one timestep ahead and with only coarse values.

        Returns:
            out_coarse (torch.LongTensor [batch_size, signal_length, ?bins?]): Predicted
                categorical distribution over ``bins`` categories for the ``coarse`` random
                variable. ? In training mode, then an extra ``bins`` dimension is included.
            out_fine (torch.LongTensor [batch_size, signal_length, *bins]): Predicted
                categorical distribution over ``bins`` categories for the ``fine`` random
                variable. ? In training mode, then an extra ``bins`` dimension is included.
            hidden (torch.FloatTensor [batch_size, size]): Final RNN hidden state.
        """
        if input_signal is not None and target_coarse is not None:
            assert input_signal.shape[1] == target_coarse.shape[
                1], 'Target signal and input signal must be of the same length'
            assert len(target_coarse.shape) == 3, (
                '``target_coarse`` must be shaped [batch_size, signal_length, 1]')
            assert len(input_signal.shape) == 3, (
                '``input_signal`` must be shaped [batch_size, signal_length, 2]')
            assert torch.min(input_signal) >= 0 and torch.max(input_signal) < self.bins
            assert torch.min(target_coarse) >= 0 and torch.max(target_coarse) < self.bins

        # [batch_size, local_length, local_features_size] →
        # [batch_size, 3, self.size, signal_length]
        conditional = self.conditional_features_upsample(local_features)

        # [batch_size, 3, self.size, signal_length] →
        # [batch_size, signal_length, 3, self.size]
        conditional = conditional.permute(0, 3, 1, 2)

        if input_signal is not None and target_coarse is not None:
            assert conditional.shape[1] == input_signal.shape[1], (
                'Upsampling parameters in tangent with signal shape and local features shape ' +
                'must be the same length after upsampling.')
            input_signal = self._scale(input_signal)
            target_coarse = self._scale(target_coarse)
            return self.train_forward(conditional, input_signal, target_coarse)

        return self.inference_forward(conditional)

    def train_forward(self, conditional, input_signal, target_coarse):
        """ Run WaveRNN in training mode with teacher-forcing.

        Args:
            conditional (torch.FloatTensor
                [batch_size, signal_length, 3, self.size]): Features to condition signal
                generation.
            input_signal (torch.FloatTensor [batch_size, signal_length, 2]): Course
                ``signal[:, :, 0]`` and fines values ``signal[:, :, 1]`` used for teacher forcing
                each between the range [-1, 1].
            target_coarse (torch.FloatTensor [batch_size, signal_length, 1]): Same as the input
                signal but one timestep ahead and with only coarse values.

        Returns:
            out_coarse (torch.LongTensor [batch_size, signal_length, bins]): Predicted
                categorical distribution over ``bins`` categories for the ``coarse`` random
                variable.
            out_fine (torch.LongTensor [batch_size, signal_length, bins]): Predicted
                categorical distribution over ``bins`` categories for the ``fine`` random
                variable.
            hidden (torch.FloatTensor [batch_size, size]): Final RNN hidden state.
        """
        batch_size, signal_length, _ = input_signal.shape

        # [batch_size, signal_length, 2] → [batch_size, signal_length, 3 * self.half_size]
        coarse_input_projection = self.project_coarse_input(input_signal)
        # [batch_size, signal_length, 3 * self.half_size] →
        # [batch_size, signal_length, 3, self.half_size]
        coarse_input_projection = coarse_input_projection.view(batch_size, signal_length, 3,
                                                               self.half_size)

        # fine_input [batch_size, signal_length, 3]
        fine_input = torch.cat([input_signal, target_coarse], dim=2)
        # [batch_size, signal_length, 3] → [batch_size, signal_length, 3 * self.half_size]
        fine_input_projection = self.project_fine_input(fine_input)
        # [batch_size, signal_length, 3 * self.half_size] →
        # [batch_size, signal_length, 3, self.half_size]
        fine_input_projection = fine_input_projection.view(batch_size, signal_length, 3,
                                                           self.half_size)

        # [batch_size, signal_length, 3, self.half_size] →
        # [batch_size, signal_length, 3, self.size]
        rnn_input = torch.cat((coarse_input_projection, fine_input_projection), dim=3)
        rnn_input += conditional
        # [batch_size, signal_length, 3, self.size] →
        # [batch_size, signal_length, 3 * self.size]
        rnn_input = rnn_input.view(batch_size, signal_length, 3 * self.size)

        # [batch_size, signal_length, 3 * self.size] →
        # [signal_length, batch_size, 3 * self.size]
        rnn_input = rnn_input.transpose(0, 1)

        # [signal_length, batch_size, 3 * self.size] →
        # hidden_states [signal_length, batch_size, self.size]
        hidden_states, last_hidden = self.stripped_gru(rnn_input)

        # [signal_length, batch_size, self.size] →
        # [batch_size, signal_length, self.size]
        hidden_states = hidden_states.transpose(0, 1)

        # [batch_size, signal_length, self.half_size]
        hidden_coarse, hidden_fine = torch.split(hidden_states, self.half_size, dim=2)

        # [batch_size, signal_length, self.half_size] → [batch_size, signal_length, bins]
        out_coarse = self.to_bins_coarse(hidden_coarse)
        out_fine = self.to_bins_fine(hidden_coarse)

        return out_coarse, out_fine, last_hidden.squeeze(0)

    def _initial_state(self, reference, batch_size):
        """ Initial state returns the initial hidden state and go sample.

        Args:
            reference (torch.Tensor): Tensor to reference device and dtype from.
            batch_size (int): Size of the batch

        Return:
            coarse (torch.FloatTensor [batch_size, 1]): Initial coarse value from [-1, 1]
            fine (torch.FloatTensor [batch_size, 1]): Initial fine value from [-1, 1]
            coarse_last_hidden (torch.FloatTensor [batch_size, self.half_size]): Initial RNN hidden
                state.
            fine_last_hidden (torch.FloatTensor [batch_size, self.half_size]): Initial RNN hidden
                state.
        """
        # Set the split signal value at signal value zero
        coarse, fine = split_signal(reference.new_zeros(batch_size))
        coarse = self._scale(coarse.unsqueeze(1))
        fine = self._scale(fine.unsqueeze(1))
        coarse_last_hidden = reference.new_zeros(batch_size, self.half_size)
        fine_last_hidden = reference.new_zeros(batch_size, self.half_size)
        return coarse, fine, coarse_last_hidden, fine_last_hidden

    def inference_forward(self, conditional):
        """  Run WaveRNN in inference mode.

        TODO:
            * Add stripped GRU replacing the sigmoid and tanh computations

        Variables:
            r: ``r`` stands for reset gate
            u: ``u`` stands for update gate
            e: ``e`` stands for memory

        Reference:
            * PyTorch GRU Equations
              https://pytorch.org/docs/stable/nn.html?highlight=gru#torch.nn.GRU
            * Efficient Neural Audio Synthesis Equations
              https://arxiv.org/abs/1802.08435

        Args:
            conditional (torch.FloatTensor
                [batch_size, signal_length, 3, self.size]): Features to condition signal
                generation.

        Returns:
            out_coarse (torch.LongTensor [batch_size, signal_length, bins]): Predicted
                categorical distribution over ``bins`` categories for the ``coarse`` random
                variable.
            out_fine (torch.LongTensor [batch_size, signal_length, bins]): Predicted
                categorical distribution over ``bins`` categories for the ``fine`` random
                variable.
            hidden (torch.FloatTensor [batch_size, size]): Final RNN hidden state.
        """
        # Some initial parameters
        batch_size, signal_length, _, _ = conditional.shape
        weight_hh_l0 = self.stripped_gru.gru.weight_hh_l0.detach()
        bias_hh_l0 = self.stripped_gru.gru.bias_hh_l0.detach()

        # Create linear projection similar to GRU
        project_hidden = nn.Linear(self.size, 3 * self.size)
        project_hidden = project_hidden.to(device=conditional.device)
        project_hidden.weight = nn.Parameter(weight_hh_l0, requires_grad=False)
        project_hidden.bias = nn.Parameter(bias_hh_l0, requires_grad=False)

        # [size * 3] → ... [half_size]
        bias = self.stripped_gru.gru.bias_ih_l0.detach()
        bias_r, bias_u, bias_e = bias.chunk(3)

        # Bias conditional
        conditional[:, :, 0] += bias_r
        conditional[:, :, 1] += bias_u
        conditional[:, :, 2] += bias_e

        # ... [batch_size, signal_length, half_size]
        conditional_coarse_r, conditional_fine_r = torch.chunk(conditional[:, :, 0], 2, dim=2)
        conditional_coarse_u, conditional_fine_u = torch.chunk(conditional[:, :, 1], 2, dim=2)
        conditional_coarse_e, conditional_fine_e = torch.chunk(conditional[:, :, 2], 2, dim=2)

        # Initial inputs
        out_coarse, out_fine = [], []
        coarse, fine, coarse_last_hidden, fine_last_hidden = self._initial_state(
            conditional, batch_size)
        for i in tqdm(range(signal_length)):
            # [batch_size, size]
            hidden = torch.cat((coarse_last_hidden, fine_last_hidden), dim=1)
            # [batch_size, size] → [batch_size, 3 * size]
            hidden_projection = project_hidden(hidden)
            # [batch_size, 3 * size] → ... [batch_size, half_size]
            (hidden_coarse_r, hidden_fine_r, hidden_coarse_u, hidden_fine_u, hidden_coarse_e,
             hidden_fine_e) = hidden_projection.chunk(
                 6, dim=1)

            del hidden_projection

            # coarse_input [batch_size, 2]
            coarse_input = torch.cat([coarse, fine], dim=1)
            assert torch.min(coarse_input) <= 1 and torch.max(coarse_input) >= -1
            # [batch_size, 2] → [batch_size, 3 * half_size]
            coarse_projection = self.project_coarse_input(coarse_input)
            # [batch_size, half_size]
            coarse_r, coarse_u, coarse_e = coarse_projection.chunk(3, dim=1)

            del coarse_projection

            # Compute the coarse gates [batch_size, half_size]
            r = sigmoid(hidden_coarse_r + coarse_r + conditional_coarse_r[:, i])
            u = sigmoid(hidden_coarse_u + coarse_u + conditional_coarse_u[:, i])
            e = tanh(r * hidden_coarse_e + coarse_e + conditional_coarse_e[:, i])
            coarse_last_hidden = u * coarse_last_hidden + (1.0 - u) * e

            del r, coarse_r, hidden_coarse_r
            del u, coarse_u, hidden_coarse_u
            del e, coarse_e, hidden_coarse_e

            # [batch_size, half_size] → [batch_size, bins]
            coarse = self.to_bins_coarse(coarse_last_hidden)
            # SOURCE: Efficient Neural Audio Synthesis
            # Once c_t has been sampled from P(c_t)
            # [batch_size, bins] → [batch_size]
            coarse = coarse.max(dim=1)[1]
            out_coarse.append(coarse)

            # Compute fine gates
            # [batch_size] → [batch_size, 1]
            coarse = self._scale(coarse).unsqueeze(1)
            # fine_input [batch_size, 3]
            fine_input = torch.cat([coarse_input, coarse], dim=1)
            assert torch.min(fine_input) <= 1 and torch.max(fine_input) >= -1
            # [batch_size, 3] → [batch_size, 3 * half_size]
            fine_projection = self.project_fine_input(fine_input)
            # ... [batch_size, half_size]
            fine_r, fine_u, fine_e = fine_projection.chunk(3, dim=1)

            del fine_projection
            del coarse_input

            # Compute the fine gates [batch_size, half_size]
            r = sigmoid(hidden_fine_r + fine_r + conditional_fine_r[:, i])
            u = sigmoid(hidden_fine_u + fine_u + conditional_fine_u[:, i])
            e = tanh(r * hidden_fine_e + fine_e + conditional_fine_e[:, i])
            fine_last_hidden = u * fine_last_hidden + (1.0 - u) * e

            del r, fine_r, hidden_fine_r
            del u, fine_u, hidden_fine_u
            del e, fine_e, hidden_fine_e

            # Compute the fine output
            # [batch_size, half_size] → [batch_size, bins]
            fine = self.to_bins_fine(fine_last_hidden)

            # SOURCE: Efficient Neural Audio Synthesis
            # Once ct has been sampled from P(ct), the gates are evaluated for the fine bits and
            # ft is sampled.
            # [batch_size, bins] → [batch_size]
            fine = fine.max(dim=1)[1]
            out_fine.append(fine)
            # [batch_size] → [batch_size, 1]
            fine = self._scale(fine).unsqueeze(1)

        # TODO: Try embeddings for WaveRNN instead of scalar values

        hidden = torch.cat((coarse_last_hidden, fine_last_hidden), dim=1)
        return torch.stack(out_coarse, dim=1), torch.stack(out_fine, dim=1), hidden

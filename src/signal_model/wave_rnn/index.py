import torch

from torch import nn
from torch.nn.functional import log_softmax
from torch.nn.functional import sigmoid
from torch.nn.functional import tanh
from torch.utils.cpp_extension import load
from tqdm import tqdm

from src.signal_model.upsample import ConditionalFeaturesUpsample
from src.signal_model.stripped_gru import StrippedGRU
from src.utils.configurable import configurable
from src.utils import split_signal

inference = load(name="inference", sources=["inference.cpp"])


def _scale(tensor, bins=256):
    """ Scale from [0, bins - 1] to [-1.0, 1.0].

    Args:
        tensor (torch.FloatTensor): Tensor between the range of [0, bins]
        bins (int): The maximum number of discrete values from [0, bins]

    Returns:
        tensor (torch.FloatTensor): Tensor between the range of [-1.0, 1.0]
    """
    assert torch.min(tensor) >= 0 and torch.max(tensor) < bins
    return tensor.float() / ((bins - 1) / 2) - 1.0


class _WaveRNNInference(nn.Module):

    def __init__(self,
                 gru_input_bias,
                 gru_hidden_weight,
                 gru_hidden_bias,
                 project_coarse_input_layer,
                 project_fine_input_layer,
                 to_bins_coarse_layer,
                 to_bins_fine_layer,
                 hidden_size=896,
                 bits=16,
                 argmax=False):
        super(_WaveRNNInference, self).__init__()
        assert hidden_size % 2 == 0, "Hidden size must be even."
        assert bits % 2 == 0, "Bits must be even for a double softmax"
        self.bins = int(2**(bits / 2))  # Encode ``bits`` with double softmax of ``bits / 2``
        self.size = hidden_size
        self.half_size = int(self.size / 2)

        # Output fully connected layers
        self.to_bins_coarse = to_bins_coarse_layer
        self.to_bins_fine = to_bins_fine_layer

        # Input fully connected layers
        self.project_coarse_input = project_coarse_input_layer
        self.project_fine_input = project_fine_input_layer

        self.argmax = argmax

        # Create linear projection similar to GRU
        self.project_hidden = nn.Linear(self.size, 3 * self.size)
        self.project_hidden.weight = nn.Parameter(gru_hidden_weight, requires_grad=False)
        self.project_hidden.bias = nn.Parameter(gru_hidden_bias, requires_grad=False)

        # [size * 3] → ... [size]
        self.bias_r, self.bias_u, self.bias_e = gru_input_bias.chunk(3)

    def _initial_state(self, reference, batch_size, hidden_state=None):
        """ Initial state returns the initial hidden state and go sample.

        Args:
            reference (torch.Tensor): Tensor to reference device and dtype from.
            batch_size (int): Size of the batch.
            hidden_state (torch.FloatTensor [batch_size, self.size], optional): GRU hidden state.

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
        # TODO: Coarse and fine should be part of the hidden state
        coarse = _scale(coarse.unsqueeze(1), self.bins)
        fine = _scale(fine.unsqueeze(1), self.bins)
        if hidden_state is not None:
            coarse_last_hidden, fine_last_hidden = hidden_state.chunk(2, dim=1)
        else:
            coarse_last_hidden = reference.new_zeros(batch_size, self.half_size)
            fine_last_hidden = reference.new_zeros(batch_size, self.half_size)
        return coarse, fine, coarse_last_hidden, fine_last_hidden

    def forward(self, conditional, hidden_state=None):
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
            hidden_state (torch.FloatTensor [batch_size, self.size], optional): GRU hidden state.

        Returns:
            out_coarse (torch.LongTensor [batch_size, signal_length, bins]): Predicted
                categorical distribution over ``bins`` categories for the ``coarse`` random
                variable.
            out_fine (torch.LongTensor [batch_size, signal_length, bins]): Predicted
                categorical distribution over ``bins`` categories for the ``fine`` random
                variable.
            hidden (torch.FloatTensor [batch_size, size]): Final RNN hidden state.
        """
        batch_size, signal_length, _, _ = conditional.shape

        # Bias conditional
        conditional[:, :, 0] += self.bias_r
        conditional[:, :, 1] += self.bias_u
        conditional[:, :, 2] += self.bias_e

        # ... [batch_size, signal_length, half_size]
        conditional_coarse_r, conditional_fine_r = torch.chunk(conditional[:, :, 0], 2, dim=2)
        conditional_coarse_u, conditional_fine_u = torch.chunk(conditional[:, :, 1], 2, dim=2)
        conditional_coarse_e, conditional_fine_e = torch.chunk(conditional[:, :, 2], 2, dim=2)

        # [batch_size, signal_length, half_size] → [batch_size, signal_length, 3 * half_size]
        conditional_coarse = torch.cat(
            (conditional_coarse_r, conditional_coarse_u, conditional_coarse_e), dim=2)
        conditional_fine = torch.cat(
            (conditional_fine_r, conditional_fine_u, conditional_fine_e), dim=2)

        del conditional_coarse_r, conditional_fine_r
        del conditional_coarse_u, conditional_fine_u
        del conditional_coarse_e, conditional_fine_e

        # Initial inputs
        out_coarse, out_fine = [], []
        coarse, fine, coarse_last_hidden, fine_last_hidden = self._initial_state(
            conditional, batch_size, hidden_state)
        for i in tqdm(range(signal_length)):
            # [batch_size, size]
            hidden = torch.cat((coarse_last_hidden, fine_last_hidden), dim=1)
            # [batch_size, size] → [batch_size, 3 * size]
            hidden_projection = self.project_hidden(hidden)
            # [batch_size, 3 * size] → ... [batch_size, half_size]
            (hidden_coarse_r, hidden_fine_r, hidden_coarse_u, hidden_fine_u, hidden_coarse_e,
             hidden_fine_e) = hidden_projection.chunk(
                 6, dim=1)
            # [batch_size, half_size] → [batch_size, size]
            hidden_coarse_ru = torch.cat((hidden_coarse_r, hidden_coarse_u), dim=1)
            hidden_fine_ru = torch.cat((hidden_fine_r, hidden_fine_u), dim=1)

            del hidden_projection
            del hidden_coarse_r, hidden_coarse_u
            del hidden_fine_r, hidden_fine_u

            # coarse_input [batch_size, 2]
            coarse_input = torch.cat([coarse, fine], dim=1)
            assert torch.min(coarse_input) <= 1 and torch.max(coarse_input) >= -1
            # [batch_size, 2] → [batch_size, 3 * half_size]
            coarse_projection = self.project_coarse_input(coarse_input)
            coarse_projection += conditional_coarse[:, i]

            # Compute the coarse gates [batch_size, half_size]
            # [batch_size, 3 * half_size] → [batch_size, size]
            coarse_ru = coarse_projection[:, :self.size]
            # [batch_size, 3 * half_size] → [batch_size, half_size]
            coarse_e = coarse_projection[:, self.size:]

            del coarse_projection

            ru = sigmoid(coarse_ru + hidden_coarse_ru)
            # [batch_size, size] → [batch_size, half_size]
            r, u = ru.chunk(2, dim=1)
            e = tanh(r * hidden_coarse_e + coarse_e)
            coarse_last_hidden = u * coarse_last_hidden + (1.0 - u) * e

            del ru, coarse_ru, hidden_coarse_ru, r, u
            del e, coarse_e, hidden_coarse_e

            # [batch_size, half_size] → [batch_size, bins]
            coarse = self.to_bins_coarse(coarse_last_hidden)
            # SOURCE: Efficient Neural Audio Synthesis
            # Once c_t has been sampled from P(c_t)
            # [batch_size, bins] → [batch_size]
            if self.argmax:
                coarse = coarse.max(dim=1)[1]
            else:
                posterior = log_softmax(coarse, dim=1)
                coarse = torch.distributions.Categorical(logits=posterior).sample()
            out_coarse.append(coarse)

            # Compute fine gates
            # [batch_size] → [batch_size, 1]
            coarse = _scale(coarse, self.bins).unsqueeze(1)
            # fine_input [batch_size, 3]
            fine_input = torch.cat([coarse_input, coarse], dim=1)
            assert torch.min(fine_input) <= 1 and torch.max(fine_input) >= -1
            # [batch_size, 3] → [batch_size, 3 * half_size]
            fine_projection = self.project_fine_input(fine_input)
            fine_projection += conditional_fine[:, i]

            # Compute the coarse gates [batch_size, half_size]
            # [batch_size, 3 * half_size] → [batch_size, size]
            fine_ru = fine_projection[:, :self.size]
            # [batch_size, 3 * half_size] → [batch_size, half_size]
            fine_e = fine_projection[:, self.size:]

            del fine_projection, coarse_input

            ru = sigmoid(fine_ru + hidden_fine_ru)
            # [batch_size, size] → [batch_size, half_size]
            r, u = ru.chunk(2, dim=1)
            e = tanh(r * hidden_fine_e + fine_e)
            fine_last_hidden = u * fine_last_hidden + (1.0 - u) * e

            del ru, fine_ru, hidden_fine_ru, r, u
            del e, fine_e, hidden_fine_e

            # Compute the fine output
            # [batch_size, half_size] → [batch_size, bins]
            fine = self.to_bins_fine(fine_last_hidden)
            # SOURCE: Efficient Neural Audio Synthesis
            # Once ct has been sampled from P(ct), the gates are evaluated for the fine bits and
            # ft is sampled.
            # [batch_size, bins] → [batch_size]
            if self.argmax:
                fine = fine.max(dim=1)[1]
            else:
                posterior = log_softmax(fine, dim=1)
                fine = torch.distributions.Categorical(logits=posterior).sample()
            out_fine.append(fine)
            # [batch_size] → [batch_size, 1]
            fine = _scale(fine, self.bins).unsqueeze(1)

        hidden = torch.cat((coarse_last_hidden, fine_last_hidden), dim=1)
        return torch.stack(out_coarse, dim=1), torch.stack(out_fine, dim=1), hidden


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
        local_feature_processing_layers (int): Number of Conv1D for processing the spectrogram.
        local_features_size (int): Dimensionality of local features.
        argmax (bool): During inference, sample the most likely sample or randomly based on the
            distribution.
    """

    @configurable
    def __init__(self,
                 hidden_size=896,
                 bits=16,
                 upsample_convs=[4],
                 upsample_repeat=75,
                 local_feature_processing_layers=0,
                 local_features_size=80,
                 argmax=False):
        super(WaveRNN, self).__init__()

        assert hidden_size % 2 == 0, "Hidden size must be even."
        assert bits % 2 == 0, "Bits must be even for a double softmax"
        self.bits = bits
        self.bins = int(2**(bits / 2))  # Encode ``bits`` with double softmax of ``bits / 2``
        self.size = hidden_size
        self.half_size = int(self.size / 2)

        # Output fully connected layers
        self.to_bins_coarse = nn.Sequential(
            nn.Linear(self.half_size, self.half_size), nn.ReLU(),
            nn.Linear(self.half_size, self.bins))

        self.to_bins_fine = nn.Sequential(
            nn.Linear(self.half_size, self.half_size), nn.ReLU(),
            nn.Linear(self.half_size, self.bins))

        # Input fully connected layers
        self.project_coarse_input = nn.Linear(2, 3 * self.half_size, bias=False)
        self.project_fine_input = nn.Linear(3, 3 * self.half_size, bias=False)

        self.conditional_features_upsample = ConditionalFeaturesUpsample(
            in_channels=local_features_size,
            out_channels=self.size,
            upsample_repeat=upsample_repeat,
            upsample_convs=upsample_convs,
            num_layers=3,
            upsample_chunks=1,
            local_feature_processing_layers=local_feature_processing_layers)

        self.stripped_gru = StrippedGRU(self.size)

        # Tensorflow and PyTorch initialize GRU bias differently. PyTorch uses glorot uniform for
        # GRU bias. Tensorflow uses 1.0 bias for the update and reset gate, and 0.0 for the
        # canditate gate.
        # SOURCES:
        # https://stackoverflow.com/questions/37350131/what-is-the-default-variable-initializer-in-tensorflow
        # https://github.com/tensorflow/tensorflow/blob/r1.8/tensorflow/python/ops/rnn_cell_impl.py
        # https://pytorch.org/docs/stable/_modules/torch/nn/modules/rnn.html#GRU
        # Init reset gate and update gate biases to 1
        torch.nn.init.constant_(self.stripped_gru.gru.bias_ih_l0[:self.size], 1)
        torch.nn.init.constant_(self.stripped_gru.gru.bias_hh_l0[:self.size], 1)
        # Init candidate bias to 0
        torch.nn.init.constant_(self.stripped_gru.gru.bias_ih_l0[self.size:], 0)
        torch.nn.init.constant_(self.stripped_gru.gru.bias_hh_l0[self.size:], 0)

        self.argmax = argmax

    def _export(self):
        """ Export to a kernel for inference.

        Returns:
            (torch.nn.Module): Module for running inference.
        """
        return _WaveRNNInference(
            gru_input_bias=self.stripped_gru.gru.bias_ih_l0,
            gru_hidden_weight=self.stripped_gru.gru.weight_hh_l0,
            gru_hidden_bias=self.stripped_gru.gru.bias_hh_l0,
            project_coarse_input_layer=self.project_coarse_input,
            project_fine_input_layer=self.project_fine_input,
            to_bins_coarse_layer=self.to_bins_coarse,
            to_bins_fine_layer=self.to_bins_fine,
            hidden_size=self.size,
            bits=self.bits,
            argmax=self.argmax)

    def forward(self, local_features, input_signal=None, target_coarse=None, hidden_state=None):
        """
        Args:
            local_features (torch.FloatTensor [batch_size, local_length, local_features_size]):
                Local feature to condition signal generation (e.g. spectrogram).
            input_signal (torch.FloatTensor [batch_size, signal_length, 2], optional): Course
                ``signal[:, :, 0]`` and fines values ``signal[:, :, 1]`` used for teacher forcing
                each between the range [-1, 1].
            target_coarse (torch.FloatTensor [batch_size, signal_length, 1], optional): Same as the
                input signal but one timestep ahead and with only coarse values.
            hidden_state (torch.FloatTensor [batch_size, self.size], optional): GRU hidden state.

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
            input_signal = _scale(input_signal, self.bins)
            target_coarse = _scale(target_coarse, self.bins)
            return self._train_forward(conditional, input_signal, target_coarse, hidden_state)

        kernel = self._export().to(conditional.device)
        return kernel(conditional, hidden_state)

    def _train_forward(self, conditional, input_signal, target_coarse, hidden_state=None):
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
            hidden_state (torch.FloatTensor [batch_size, self.size], optional): GRU hidden state.

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

        if hidden_state is not None:
            hidden_state = hidden_state.unsqueeze(0)

        # [signal_length, batch_size, 3 * self.size] →
        # hidden_states [signal_length, batch_size, self.size]
        hidden_states, last_hidden = self.stripped_gru(rnn_input, hidden_state)

        # [signal_length, batch_size, self.size] →
        # [batch_size, signal_length, self.size]
        hidden_states = hidden_states.transpose(0, 1)

        # [batch_size, signal_length, self.size] →
        # [batch_size, signal_length, self.half_size]
        hidden_coarse, hidden_fine = hidden_states.chunk(2, dim=2)

        # [batch_size, signal_length, self.half_size] → [batch_size, signal_length, bins]
        out_coarse = self.to_bins_coarse(hidden_coarse)
        out_fine = self.to_bins_fine(hidden_fine)
        return out_coarse, out_fine, last_hidden.squeeze(0)

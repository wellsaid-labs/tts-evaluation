import torch

from torch import nn
from torch.nn import functional
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
        self.hidden_size = hidden_size
        self.half_hidden_size = int(hidden_size / 2)

        # Output fully connected layers
        self.get_out_coarse = nn.Sequential(
            nn.Linear(self.half_hidden_size, self.half_hidden_size), nn.ReLU(),
            nn.Linear(self.half_hidden_size, self.bins))
        self.get_out_fine = nn.Sequential(
            nn.Linear(self.half_hidden_size, self.half_hidden_size), nn.ReLU(),
            nn.Linear(self.half_hidden_size, self.bins))

        # Input fully connected layers
        self.project_coarse_input = nn.Linear(2, 3 * self.half_hidden_size, bias=False)
        self.project_fine_input = nn.Linear(3, 3 * self.half_hidden_size, bias=False)

        self.conditional_features_upsample = ConditionalFeaturesUpsample(
            in_channels=local_features_size,
            out_channels=self.hidden_size,
            upsample_repeat=upsample_repeat,
            upsample_convs=upsample_convs,
            num_layers=3,  # One layer per gate (a.i. reset, memory, and update)
            upsample_chunks=1)

        self.stripped_gru = StrippedGRU(self.hidden_size)

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
            out_coarse (torch.LongTensor [batch_size, signal_length, bins]): Predicted
                categorical distribution over ``bins`` categories for the ``coarse`` random
                variable. If in training mode, then ``functional.softmax`` is not applied.
            out_fine (torch.LongTensor [batch_size, signal_length, bins]): Predicted
                categorical distribution over ``bins`` categories for the ``fine`` random
                variable. If in training mode, then ``functional.softmax`` is not applied.
        """
        if input_signal is not None and target_coarse is not None:
            assert input_signal.shape[1] == target_coarse.shape[
                1], 'Target signal and input signal must be of the same length'
            assert len(target_coarse.shape) == 3, (
                '``target_coarse`` must be shaped [batch_size, signal_length, 1]')
            assert len(input_signal.shape) == 3, (
                '``input_signal`` must be shaped [batch_size, signal_length, 2]')

        # [batch_size, local_length, local_features_size] →
        # [batch_size, 3, self.hidden_size, signal_length]
        conditional_features = self.conditional_features_upsample(local_features)

        # [batch_size, 3, self.hidden_size, signal_length] →
        # [batch_size, signal_length, 3, self.hidden_size]
        conditional_features = conditional_features.permute(0, 3, 1, 2)
        if input_signal is not None and target_coarse is not None:
            assert conditional_features.shape[1] == input_signal.shape[1], (
                'Upsampling parameters in tangent with signal shape and local features shape ' +
                'must be the same length after upsampling.')
            input_signal = self._scale(input_signal)
            target_coarse = self._scale(target_coarse)
            return self.train_forward(conditional_features, input_signal, target_coarse)

        return self.inference_forward(conditional_features)

    def train_forward(self, conditional_features, input_signal, target_coarse):
        """ Run WaveRNN in training mode with teacher-forcing.

        Args:
            conditional_features (torch.FloatTensor
                [batch_size, signal_length, 3, self.hidden_size]): Features to condition signal
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
        """
        batch_size, signal_length, _ = input_signal.shape

        # [batch_size, signal_length, 2] → [batch_size, signal_length, 3 * self.half_hidden_size]
        coarse_input_projection = self.project_coarse_input(input_signal)
        # [batch_size, signal_length, 3 * self.half_hidden_size] →
        # [batch_size, signal_length, 3, self.half_hidden_size]
        coarse_input_projection = coarse_input_projection.view(batch_size, signal_length, 3,
                                                               self.half_hidden_size)

        # fine_input [batch_size, signal_length, 3]
        fine_input = torch.cat([input_signal, target_coarse], dim=2)
        # [batch_size, signal_length, 3] → [batch_size, signal_length, 3 * self.half_hidden_size]
        fine_input_projection = self.project_fine_input(fine_input)
        # [batch_size, signal_length, 3 * self.half_hidden_size] →
        # [batch_size, signal_length, 3, self.half_hidden_size]
        fine_input_projection = fine_input_projection.view(batch_size, signal_length, 3,
                                                           self.half_hidden_size)

        # [batch_size, signal_length, 3, self.half_hidden_size] →
        # [batch_size, signal_length, 3, self.hidden_size]
        rnn_input = torch.cat((coarse_input_projection, fine_input_projection), dim=3)
        rnn_input += conditional_features
        # [batch_size, signal_length, 3, self.hidden_size] →
        # [batch_size, signal_length, 3 * self.hidden_size]
        rnn_input = rnn_input.view(batch_size, signal_length, 3 * self.hidden_size)

        # [batch_size, signal_length, 3 * self.hidden_size] →
        # [signal_length, batch_size, 3 * self.hidden_size]
        rnn_input = rnn_input.transpose(0, 1)

        # [signal_length, batch_size, 3 * self.hidden_size] →
        # [signal_length, batch_size, self.hidden_size]
        hidden_states, _ = self.stripped_gru(rnn_input)

        # [signal_length, batch_size, self.hidden_size] →
        # [batch_size, signal_length, self.hidden_size]
        hidden_states = hidden_states.transpose(0, 1)

        # [batch_size, signal_length, self.half_hidden_size]
        hidden_coarse, hidden_fine = torch.split(hidden_states, self.half_hidden_size, dim=2)

        # [batch_size, signal_length, self.half_hidden_size] → [batch_size, signal_length, bins]
        out_coarse = self.get_out_coarse(hidden_coarse)
        out_fine = self.get_out_fine(hidden_coarse)

        return out_coarse, out_fine

    def inference_forward(self, conditional_features):
        """  Run WaveRNN in inference mode.

        Args:
            conditional_features (torch.FloatTensor
                [batch_size, signal_length, 3, self.hidden_size]): Features to condition signal
                generation.

        Returns:
            out_coarse (torch.LongTensor [batch_size, signal_length, bins]): Predicted
                categorical distribution over ``bins`` categories for the ``coarse`` random
                variable.
            out_fine (torch.LongTensor [batch_size, signal_length, bins]): Predicted
                categorical distribution over ``bins`` categories for the ``fine`` random
                variable.
        """
        batch_size, signal_length, _, _ = conditional_features.shape
        device = conditional_features.device

        # ... [batch_size, signal_length, self.half_hidden_size]
        conditional_coarse_reset_gate, conditional_fine_reset_gate = torch.split(
            conditional_features[:, :, 0, :], self.half_hidden_size, dim=2)
        conditional_coarse_update_gate, conditional_fine_update_gate = torch.split(
            conditional_features[:, :, 1, :], self.half_hidden_size, dim=2)
        conditional_coarse_memory, conditional_fine_memory = torch.split(
            conditional_features[:, :, 2, :], self.half_hidden_size, dim=2)

        # ... [self.half_hidden_size]
        (bias_coarse_reset_gate, bias_fine_reset_gate, bias_coarse_update_gate,
         bias_fine_update_gate, bias_coarse_memory, bias_fine_memory) = torch.split(
             self.stripped_gru.gru.bias_ih_l0, self.half_hidden_size)

        project_hidden = nn.Linear(self.hidden_size, 3 * self.hidden_size, bias=False)
        project_hidden.weight = self.stripped_gru.gru.weight_hh_l0
        project_hidden = project_hidden.to(device=device)

        # Initial inputs
        out_coarse, out_fine = [], []
        coarse, fine = split_signal(conditional_features.new_zeros(batch_size))
        coarse = self._scale(coarse.unsqueeze(1))
        fine = self._scale(fine.unsqueeze(1))
        coarse_last_hidden = conditional_features.new_zeros(batch_size, self.half_hidden_size)
        fine_last_hidden = conditional_features.new_zeros(batch_size, self.half_hidden_size)

        for i in tqdm(range(signal_length)):
            # coarse_input [batch_size, 2]
            coarse_input = torch.cat([coarse, fine], dim=1)

            # [batch_size, 2] → [batch_size, 3 * self.half_hidden_size]
            coarse_input_projection = self.project_coarse_input(coarse_input)
            # [batch_size, self.half_hidden_size]
            coarse_input_reset_gate, coarse_input_update_gate, coarse_input_memory = \
                torch.split(coarse_input_projection, self.half_hidden_size, dim=1)

            del coarse_input_projection

            # hidden [batch_size, self.hidden_size]
            hidden = torch.cat((coarse_last_hidden, fine_last_hidden), dim=1)
            # [batch_size, self.hidden_size] → [batch_size, 3 * self.hidden_size]
            hidden = project_hidden(hidden)
            # ... [batch_size, self.half_hidden_size]
            (hidden_coarse_reset_gate, hidden_fine_reset_gate, hidden_coarse_update_gate,
             hidden_fine_update_gate, hidden_coarse_memory, hidden_fine_memory) = torch.split(
                 hidden, self.half_hidden_size, dim=1)

            # TODO: Replace with Stripped GRU
            # Compute the coarse gates
            reset_gate = functional.sigmoid(
                hidden_coarse_reset_gate + coarse_input_reset_gate + bias_coarse_reset_gate +
                conditional_coarse_reset_gate[:, i])
            update_gate = functional.sigmoid(
                hidden_coarse_update_gate + coarse_input_update_gate + bias_coarse_update_gate +
                conditional_coarse_update_gate[:, i])
            next_hidden = functional.tanh(reset_gate * hidden_coarse_memory + coarse_input_memory +
                                          bias_coarse_memory + conditional_coarse_memory[:, i])
            # hidden_coarse [batch_size, self.half_hidden_size]
            hidden_coarse = (update_gate * coarse_last_hidden + (1.0 - update_gate) * next_hidden)

            del reset_gate, coarse_input_reset_gate, hidden_coarse_reset_gate
            del update_gate, coarse_input_update_gate, hidden_coarse_update_gate
            del next_hidden, coarse_input_memory, hidden_coarse_memory

            # Compute the coarse output
            # [batch_size, self.half_hidden_size] → [batch_size, bins]
            coarse = self.get_out_coarse(hidden_coarse)
            coarse = functional.softmax(coarse, dim=1)
            out_coarse.append(coarse)

            # [batch_size, bins] → [batch_size]
            coarse = coarse.max(dim=1)[1]

            coarse = self._scale(coarse)
            # [batch_size] → [batch_size, 1]
            coarse = coarse.unsqueeze(1)

            # fine_input [batch_size, 3]
            fine_input = torch.cat([coarse_input, coarse], dim=1)
            # [batch_size, 3] → [batch_size, 3 * self.half_hidden_size]
            fine_input_projection = self.project_fine_input(fine_input)
            # ... [batch_size, self.half_hidden_size]
            fine_input_reset_gate, fine_input_update_gate, fine_input_memory = \
                torch.split(fine_input_projection, self.half_hidden_size, dim=1)

            del fine_input_projection
            del coarse_input

            # TODO: Replace with Stripped GRU
            # Compute the fine gates
            reset_gate = functional.sigmoid(
                hidden_fine_reset_gate + fine_input_reset_gate + bias_fine_reset_gate +
                conditional_fine_reset_gate[:, i])
            update_gate = functional.sigmoid(
                hidden_fine_update_gate + fine_input_update_gate + bias_fine_update_gate +
                conditional_fine_update_gate[:, i])
            next_hidden = functional.tanh(reset_gate * hidden_fine_memory + fine_input_memory +
                                          bias_fine_memory + conditional_fine_memory[:, i])
            # hidden_fine [batch_size, self.half_hidden_size]
            hidden_fine = update_gate * fine_last_hidden + (1.0 - update_gate) * next_hidden

            del reset_gate, fine_input_reset_gate, hidden_fine_reset_gate
            del update_gate, fine_input_update_gate, hidden_fine_update_gate
            del next_hidden, fine_input_memory, hidden_fine_memory

            # Compute the fine output
            # [batch_size, self.half_hidden_size] → [batch_size, bins]
            fine = self.get_out_fine(hidden_fine)
            fine = functional.softmax(fine, dim=1)
            out_fine.append(fine)

            # [batch_size, bins] → [batch_size]
            fine = fine.max(dim=1)[1]
            fine = self._scale(fine)
            # [batch_size] → [batch_size, 1]
            fine = fine.unsqueeze(1)

            coarse_last_hidden = hidden_coarse
            fine_last_hidden = hidden_fine

            del hidden_fine
            del hidden_coarse

        out_coarse = torch.stack(out_coarse, dim=1)
        out_fine = torch.stack(out_fine, dim=1)
        return out_coarse, out_fine

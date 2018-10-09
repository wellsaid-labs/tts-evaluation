import torch

from torch import nn
from torch.nn.functional import log_softmax
from tqdm import tqdm

from src.signal_model.upsample import ConditionalFeaturesUpsample
from src.signal_model.stripped_gru import StrippedGRU
from src.utils.configurable import configurable
from src.utils import split_signal


def _scale(tensor, bins=256):
    """ Scale from [0, bins - 1] to [-1.0, 1.0].

    Args:
        tensor (torch.FloatTensor): Tensor between the range of [0, bins]
        bins (int): The maximum number of discrete values from [0, bins]

    Returns:
        tensor (torch.FloatTensor): Tensor between the range of [-1.0, 1.0]
    """
    assert torch.min(tensor) >= 0 and torch.max(tensor) < bins
    return tensor.float() / ((bins - 1) / 2.0) - 1.0


class WaveRNN(nn.Module):
    """ WaveRNN as defined in "Efficient Neural Audio Synthesis".

    References:
        * Efficient Neural Audio Synthesis
          https://arxiv.org/pdf/1802.08435.pdf

    Args:
        hidden_size (int): GRU hidden state size.
        bits (int): Number of bits  Number of categories to predict for coarse and fine variables.
        upsample_num_filters (int): Filters to be used with each upsampling kernel. The last kernel
            is used for upsampling the length.
        upsample_kernels (list of tuples): Sizes of kernels used for upsampling, every kernel has an
            associated number of filters.
        upsample_repeat (int): Number of times to repeat frames.
        local_features_size (int): Dimensionality of local features.
    """

    @configurable
    def __init__(self,
                 hidden_size=896,
                 bits=16,
                 upsample_num_filters=[64, 64, 32, 10],
                 upsample_kernels=[(5, 5), (3, 3), (3, 3), (3, 3)],
                 upsample_repeat=25,
                 local_features_size=128):
        super(WaveRNN, self).__init__()

        assert hidden_size % 2 == 0, "Hidden size must be even."
        assert bits % 2 == 0, "Bits must be even for a double softmax"
        self.bits = bits
        self.bins = int(2**(bits / 2))  # Encode ``bits`` with double softmax of ``bits / 2``
        self.size = hidden_size
        self.half_size = int(self.size / 2)

        # Output fully connected layers
        gain = torch.nn.init.calculate_gain('relu')
        self.to_bins_coarse = nn.Sequential(
            nn.Linear(self.half_size, self.half_size), nn.ReLU(),
            nn.Linear(self.half_size, self.bins))
        torch.nn.init.orthogonal_(self.to_bins_coarse[0].weight, gain=gain)

        self.to_bins_fine = nn.Sequential(
            nn.Linear(self.half_size, self.half_size), nn.ReLU(),
            nn.Linear(self.half_size, self.bins))
        torch.nn.init.orthogonal_(self.to_bins_fine[0].weight, gain=gain)

        # Input fully connected layers
        self.project_coarse_input = nn.Linear(2, 3 * self.half_size, bias=False)
        self.project_fine_input = nn.Linear(3, 3 * self.half_size, bias=False)

        self.conditional_features_upsample = ConditionalFeaturesUpsample(
            in_channels=local_features_size,
            out_channels=self.size * 3,
            num_filters=upsample_num_filters,
            upsample_repeat=upsample_repeat,
            kernels=upsample_kernels)

        self.stripped_gru = StrippedGRU(self.size)

        # Orthogonal initialization for each GRU gate, following guidance from:
        # * https://smerity.com/articles/2016/orthogonal_init.html
        # * https://gist.github.com/kaniblu/81828dfcf5cca60ae93f4d7bd19aeac5
        # * https://web.stanford.edu/class/cs224n/lectures/lecture9.pdf
        # * https://hjweide.github.io/orthogonal-initialization-in-convolutional-layers
        # weight_hh_l0 [size * 3, size]
        torch.nn.init.orthogonal_(self.stripped_gru.gru.weight_hh_l0[0:self.size])
        torch.nn.init.orthogonal_(self.stripped_gru.gru.weight_hh_l0[self.size:-self.size])
        torch.nn.init.orthogonal_(self.stripped_gru.gru.weight_hh_l0[-self.size:])
        torch.nn.init.constant_(self.stripped_gru.gru.bias_ih_l0, 0)
        torch.nn.init.constant_(self.stripped_gru.gru.bias_hh_l0, 0)

    def _reorder_gru_weights(self, weight):
        """ Reorder weights to be more efficient for inference.

        Args:
            weight (torch.Tensor [3 * self.size, ...])

        Returns:
            weight (torch.Tensor [3 * self.size, ...])
        """
        (hidden_coarse_r, hidden_fine_r, hidden_coarse_u, hidden_fine_u, hidden_coarse_e,
         hidden_fine_e) = weight.chunk(
             6, dim=0)
        return torch.cat(
            (hidden_coarse_r, hidden_coarse_u, hidden_coarse_e, hidden_fine_r, hidden_fine_u,
             hidden_fine_e),
            dim=0)

    def _infer_initial_state(self, reference, batch_size, hidden_state=None):
        """ Initial state returns the initial hidden state and go sample.

        Args:
            reference (torch.Tensor): Tensor to reference device and dtype from.
            batch_size (int): Size of the batch.
            hidden_state (torch.FloatTensor [batch_size, self.size], optional): GRU hidden state.

        Return:
            coarse (torch.LongTensor [batch_size]): Initial coarse value from [0, 255]
            fine (torch.LongTensor [batch_size]): Initial fine value from [0, 255]
            coarse_last_hidden (torch.FloatTensor [self.half_size, batch_size]): Initial RNN hidden
                state.
            fine_last_hidden (torch.FloatTensor [self.half_size, batch_size]): Initial RNN hidden
                state.
        """
        if hidden_state is not None:
            assert len(hidden_state) == 4
            return hidden_state

        # Set the split signal value at signal value zero
        coarse, fine = split_signal(reference.new_zeros(batch_size))
        coarse_last_hidden = reference.new_zeros(self.half_size, batch_size)
        fine_last_hidden = reference.new_zeros(self.half_size, batch_size)
        return coarse.long(), fine.long(), coarse_last_hidden, fine_last_hidden

    def _infer_step(self,
                    last_hidden_state,
                    input_projection,
                    hidden_projection,
                    to_bins,
                    temperature=1.0,
                    argmax=False):
        """ Run a single step of Wave RNN inference.

        Args:
            last_hidden_state (torch.FloatTensor [half_size, batch_size]): Last GRU hidden state.
            input_projection (torch.FloatTensor [3 * half_size, batch_size]): Input projected into
                GRU gates r, u and e.
            hidden_projection (torch.FloatTensor [3 * half_size, batch_size]): Hidden state
                projected into GRU gates r, u and e.
            to_bins (callable): Callabe that takes [batch_size, half_size] and returns scores
                [batch_size, bins] over some bins.
            temperature (float): Passed on from ``infer``.
            argmax (bool): Passed on from ``infer``.

        Returns:
            last_hidden_state (torch.FloatTensor [half_size, batch_size]): Last GRU hidden state.
            sample (torch.LongTensor [batch_size]): Predicted batch classes.
        """
        # ru [size, batch_size]
        ru = (hidden_projection[:self.size] + input_projection[:self.size]).sigmoid()
        # [size, batch_size] → [half_size, batch_size]
        r, u = ru.chunk(2, dim=0)
        # e [half_size, batch_size]
        e = (r * hidden_projection[self.size:] + input_projection[self.size:]).tanh()
        # [half_size, batch_size]
        last_hidden_state = u * last_hidden_state + (1.0 - u) * e

        # [half_size, batch_size] → [batch_size, bins]
        bins = to_bins(last_hidden_state.transpose(0, 1))
        # [batch_size, bins] → [batch_size]
        if argmax:
            sample = bins.max(dim=1)[1]
        else:
            posterior = log_softmax(bins / temperature, dim=1)
            sample = torch.distributions.Categorical(logits=posterior).sample()

        return last_hidden_state, sample

    @configurable
    def infer(self, local_features, argmax=False, temperature=1.0, hidden_state=None, pad=True):
        """  Run WaveRNN in inference mode.

        Variables:
            r: ``r`` stands for reset gate
            u: ``u`` stands for update gate
            e: ``e`` stands for memory

        Reference: # noqa
            * PyTorch GRU Equations
              https://pytorch.org/docs/stable/nn.html?highlight=gru#torch.nn.GRU
            * Efficient Neural Audio Synthesis Equations
              https://arxiv.org/abs/1802.08435
            * https://cs.stackexchange.com/questions/79241/what-is-temperature-in-lstm-and-neural-networks-generally

        Args:
            local_features (torch.FloatTensor [batch_size, local_length, local_features_size]):
                Local feature to condition signal generation (e.g. spectrogram).
            argmax (bool, optional): If ``True``, during inference sample the most likely sample.
            temperature (float, optional): Temperature to control the variance in softmax
                predictions.
            hidden_state (tuple, optional): Initial hidden state with RNN hidden state and last
                coarse/fine samples.
            pad (bool, optional): Pad the spectrogram with zeros on the ends, assuming that the
                spectrogram has no context on the ends.

        Returns:
            out_coarse (torch.LongTensor [batch_size, signal_length]): Predicted
                categorical distribution over ``bins`` categories for the ``coarse`` random
                variable.
            out_fine (torch.LongTensor [batch_size, signal_length]): Predicted
                categorical distribution over ``bins`` categories for the ``fine`` random
                variable.
            hidden_state (tuple): Hidden state with RNN hidden state and last coarse/fine samples.
        """
        # [batch_size, local_length, local_features_size] →
        # [batch_size, self.size * 3, signal_length]
        conditional = self.conditional_features_upsample(local_features, pad=pad)

        batch_size, _, signal_length = conditional.shape

        # [batch_size, self.size * 3, signal_length] →
        # [batch_size, signal_length, self.size * 3] →
        conditional = conditional.transpose(1, 2)

        # [batch_size, signal_length,  3 * self.size] →
        # [batch_size, signal_length,  3, self.size]
        conditional = conditional.view(batch_size, signal_length, 3, self.size)

        # [size * 3] → bias_r, bias_u, bias_e [size]
        bias_r, bias_u, bias_e = self.stripped_gru.gru.bias_ih_l0.chunk(3)

        # ... [batch_size, signal_length, half_size]
        bias_coarse_r, bias_fine_r = torch.chunk(conditional[:, :, 0] + bias_r, 2, dim=2)
        bias_coarse_u, bias_fine_u = torch.chunk(conditional[:, :, 1] + bias_u, 2, dim=2)
        bias_coarse_e, bias_fine_e = torch.chunk(conditional[:, :, 2] + bias_e, 2, dim=2)

        # [batch_size, signal_length, half_size] → [batch_size, signal_length, 3 * half_size]
        bias_coarse = torch.cat((bias_coarse_r, bias_coarse_u, bias_coarse_e), dim=2)
        bias_fine = torch.cat((bias_fine_r, bias_fine_u, bias_fine_e), dim=2)

        del bias_coarse_r, bias_fine_r
        del bias_coarse_u, bias_fine_u
        del bias_coarse_e, bias_fine_e

        # [batch_size, signal_length, 3 * half_size] → [3 * half_size, batch_size, signal_length]
        bias_coarse = bias_coarse.transpose(0, 2)
        bias_fine = bias_fine.transpose(0, 2)

        project_hidden_weights = self._reorder_gru_weights(self.stripped_gru.gru.weight_hh_l0)
        project_hidden_bias = self._reorder_gru_weights(
            self.stripped_gru.gru.bias_hh_l0).unsqueeze(1)

        # Initial inputs
        out_coarse, out_fine = [], []
        coarse, fine, coarse_last_hidden, fine_last_hidden = self._infer_initial_state(
            conditional, batch_size, hidden_state)
        for i in tqdm(range(signal_length)):
            # [size, batch_size]
            hidden = torch.cat((coarse_last_hidden, fine_last_hidden), dim=0)
            # [3 * size] + [3 * size, size] * [size, batch_size] = [3 * size, batch_size]
            hidden_projection = torch.addmm(project_hidden_bias, project_hidden_weights, hidden)

            # coarse_input [2, batch_size]
            coarse_input = torch.stack([coarse, fine], dim=0)
            # [3 * half_size, batch_size] + [3 * half_size, 2] * [2, batch_size] →
            # [3 * half_size, batch_size]
            coarse_projection = torch.addmm(bias_coarse[:, i], self.project_coarse_input.weight,
                                            _scale(coarse_input))
            coarse_last_hidden, coarse = self._infer_step(
                coarse_last_hidden,
                coarse_projection,
                hidden_projection[:3 * self.half_size],
                self.to_bins_coarse,
                temperature=temperature,
                argmax=argmax)
            out_coarse.append(coarse)

            # cat([2, batch_size], [1, batch_size]) → [3, batch_size]
            fine_input = torch.cat([coarse_input, coarse.unsqueeze(0)], dim=0)
            # [3 * half_size, batch_size] + [3 * half_size, 2] * [2, batch_size] →
            # [3 * half_size, batch_size]
            fine_projection = torch.addmm(bias_fine[:, i], self.project_fine_input.weight,
                                          _scale(fine_input))
            fine_last_hidden, fine = self._infer_step(
                fine_last_hidden,
                fine_projection,
                hidden_projection[3 * self.half_size:],
                self.to_bins_fine,
                temperature=temperature,
                argmax=argmax)
            out_fine.append(fine)

        out_coarse = torch.stack(out_coarse, dim=1)
        out_fine = torch.stack(out_fine, dim=1)
        hidden_state = (coarse, fine, coarse_last_hidden, fine_last_hidden)
        return out_coarse, out_fine, hidden_state

    def forward(self, local_features, input_signal, target_coarse, hidden_state=None, pad=False):
        """
        Args:
            local_features (torch.FloatTensor [batch_size, local_length, local_features_size]):
                Local feature to condition signal generation (e.g. spectrogram).
            input_signal (torch.FloatTensor [batch_size, signal_length, 2], optional): Course
                ``signal[:, :, 0]`` and fines values ``signal[:, :, 1]`` used for teacher forcing
                each between the range [-1, 1].
            target_coarse (torch.FloatTensor [batch_size, signal_length, 1], optional): Same as the
                input signal but one timestep ahead and with only coarse values.
            hidden_state (torch.FloatTensor [batch_size, size], optional): Initial GRU hidden state.
            pad (bool, optional): Pad the spectrogram with zeros on the ends, assuming that the
                spectrogram has no context on the ends.

        Returns:
            out_coarse (torch.LongTensor [batch_size, signal_length, bins]): Predicted
                categorical distribution over ``bins`` categories for the ``coarse`` random
                variable.
            out_fine (torch.LongTensor [batch_size, signal_length, bins]): Predicted
                categorical distribution over ``bins`` categories for the ``fine`` random
                variable.
            hidden_state (torch.FloatTensor [batch_size, size], optional): Final GRU hidden state.
        """
        assert input_signal.shape[1] == target_coarse.shape[
            1], 'Target signal and input signal must be of the same length'
        assert len(target_coarse.shape) == 3, (
            '``target_coarse`` must be shaped [batch_size, signal_length, 1]')
        assert len(input_signal.shape) == 3, (
            '``input_signal`` must be shaped [batch_size, signal_length, 2]')

        # [batch_size, local_length, local_features_size] →
        # [batch_size, 3 * self.size, signal_length]
        conditional = self.conditional_features_upsample(local_features, pad=pad)

        # [batch_size, 3 * self.size, signal_length] →
        # [batch_size, signal_length,  3 * self.size]
        conditional = conditional.transpose(1, 2)

        batch_size, signal_length, _ = conditional.shape

        # [batch_size, signal_length,  3 * self.size] →
        # [batch_size, signal_length,  3, self.size]
        conditional = conditional.view(batch_size, signal_length, 3, self.size)

        assert conditional.shape[1] == input_signal.shape[1], (
            'Upsampling parameters in tangent with signal shape and local features shape ' +
            'must be the same length after upsampling.')
        input_signal = _scale(input_signal, self.bins)
        target_coarse = _scale(target_coarse, self.bins)

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

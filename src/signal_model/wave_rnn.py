from torch import nn
from torch.jit import script
from torch.jit import script_method
from torch.jit import ScriptModule
from torch.jit import trace
from torch.nn.functional import softmax

import torch

from src.audio import split_signal
from src.utils import log_runtime
from src.hparams import configurable
from src.hparams import ConfiguredArg
from src.signal_model.stripped_gru import StrippedGRU
from src.signal_model.upsample import ConditionalFeaturesUpsample


@script
def _gru_gates(hidden_r, hidden_i, hidden_n, input, last_hidden):
    """ Compute GRU gates.

    Variables:
        r: ``r`` stands for reset gate
        i: ``i`` stands for input gate
        n: ``n`` stands for new gate

    Args:
        hidden_r (torch.FloatTensor [hidden_size, batch_size])
        hidden_i (torch.FloatTensor [hidden_size, batch_size])
        hidden_n (torch.FloatTensor [hidden_size, batch_size])
        input (torch.FloatTensor [3 * hidden_size, batch_size])
        last_hidden (torch.FloatTensor [hidden_size, batch_size])

    Returns:
        torch.FloatTensor [hidden_size, batch_size]
    """
    input_r, input_i, input_n = input.chunk(3)

    reset_gate = torch.sigmoid(input_r + hidden_r)
    input_gate = torch.sigmoid(input_i + hidden_i)
    new_gate = torch.tanh(input_n + reset_gate * hidden_n)
    return new_gate + input_gate * (last_hidden - new_gate)


class _WaveRNNInferrer(ScriptModule):
    """ WaveRNN JIT ScriptModule for computing inference.

    Args:
        hidden_size (int): GRU hidden state size.
        bits (int): Number of bits  Number of categories to predict for coarse and fine variables.
        to_bins_coarse (torch.nn.Module): Provided by WaveRNN.
        to_bins_fine (torch.nn.Module): Provided by WaveRNN.
        project_coarse_input (torch.nn.Module): Provided by WaveRNN.
        project_fine_input (torch.nn.Module): Provided by WaveRNN.
        conditional_features_upsample (torch.nn.Module): Provided by WaveRNN.
        stripped_gru (torch.nn.Module): Provided by WaveRNN.
        argmax (bool, optional): If ``True``, during sampling pick the most likely bin instead of
            sampling from a multinomial distribution.
        device (torch.device, optional): Device this inference module is to run on.
    """
    __constants__ = ['size', 'half_size', 'inverse_bins']

    def __init__(self,
                 hidden_size,
                 bits,
                 to_bins_coarse,
                 to_bins_fine,
                 project_coarse_input,
                 project_fine_input,
                 conditional_features_upsample,
                 stripped_gru,
                 argmax=False,
                 device=torch.device('cpu')):
        super(_WaveRNNInferrer, self).__init__()
        self.bits = bits
        self.bins = int(2**(bits / 2))  # Encode ``bits`` with double softmax of ``bits / 2``
        self.inverse_bins = 1 / ((self.bins - 1) / 2)
        self.size = hidden_size
        self.half_size = int(self.size / 2)
        self.argmax = argmax

        # Output fully connected layers
        self.to_bins_coarse = to_bins_coarse
        self.to_bins_fine = to_bins_fine

        # Input fully connected layers
        self.project_coarse_input = project_coarse_input
        self.project_fine_input = project_fine_input

        self.conditional_features_upsample = conditional_features_upsample
        self.stripped_gru = stripped_gru

        # NOTE: Move to device before tracing because tracing tends to 'lock-in' a device.
        if device.type == 'cuda':
            self.cuda(device)
        elif device.type == 'cpu':
            self.cpu()

        self.eval()  # Evaluation mode by default

        self._init_infer_step_traced(device=device)

    def _init_infer_step_traced(self, device, trace_batch_size=2):
        """ Trace ``_infer_step`` into with JIT """

        # Dumy inputs for traciing
        bins = self.bins
        get_project_input_bias = lambda: torch.rand(3 * self.half_size, trace_batch_size).to(device)
        project_hidden_weights = self._reorder_gru_weights(
            self.stripped_gru.gru.weight_hh_l0).to(device)
        project_hidden_bias = self._reorder_gru_weights(
            self.stripped_gru.gru.bias_hh_l0).unsqueeze(1).to(device)
        get_last_output = lambda: torch.LongTensor(trace_batch_size).random_(0, bins).to(device)
        get_last_hidden = lambda: torch.rand(self.half_size, trace_batch_size).to(device)

        # JIT trace
        self._infer_step_traced = trace(
            self._infer_step, (get_last_output(), get_last_hidden(), get_last_output(),
                               get_last_hidden(), project_hidden_bias, project_hidden_weights,
                               get_project_input_bias(), get_project_input_bias()),
            check_trace=False)

    def _reorder_gru_weights(self, weight):
        """ Reorder weights to be more efficient for inference.

        Args:
            weight (torch.Tensor [3 * self.size, ...])

        Returns:
            weight (torch.Tensor [3 * self.size, ...])
        """
        (hidden_coarse_r, hidden_fine_r, hidden_coarse_u, hidden_fine_u, hidden_coarse_e,
         hidden_fine_e) = weight.chunk(6)
        return torch.cat((hidden_coarse_r, hidden_coarse_u, hidden_coarse_e, hidden_fine_r,
                          hidden_fine_u, hidden_fine_e))

    def _infer_step(self, coarse_last, coarse_last_hidden, fine_last, fine_last_hidden,
                    project_hidden_bias, project_hidden_weights, project_coarse_bias,
                    project_fine_bias):
        # [size, batch_size]
        hidden = torch.cat((coarse_last_hidden, fine_last_hidden))
        # [3 * size] + [3 * size, size] * [size, batch_size] = [3 * size, batch_size]
        hidden_projection = torch.addmm(project_hidden_bias, project_hidden_weights, hidden)
        (coarse_hidden_r, coarse_hidden_i, coarse_hidden_n, fine_hidden_r, fine_hidden_i,
         fine_hidden_n) = hidden_projection.chunk(6)

        # coarse_input [2, batch_size]
        coarse_input = torch.stack([coarse_last, fine_last]).float() * self.inverse_bins - 1
        # [3 * half_size, batch_size] + [3 * half_size, 2] * [2, batch_size] →
        # [3 * half_size, batch_size]
        coarse_input_projection = torch.addmm(project_coarse_bias, self.project_coarse_input.weight,
                                              coarse_input)
        coarse_last_hidden = _gru_gates(coarse_hidden_r, coarse_hidden_i, coarse_hidden_n,
                                        coarse_input_projection, coarse_last_hidden)

        # Predict
        # [half_size, batch_size] → [batch_size, bins]
        coarse_bins = self.to_bins_coarse(coarse_last_hidden.transpose(0, 1))
        # [batch_size, bins] → [batch_size]
        if self.argmax:
            coarse_last = coarse_bins.max(dim=1)[1]
        else:
            coarse_posterior = softmax(coarse_bins, dim=1)
            coarse_last = torch.multinomial(coarse_posterior, 1).squeeze(1)

        # cat([2, batch_size], [1, batch_size]) → [3, batch_size]
        scaled_coarse_last = coarse_last.float().unsqueeze(0) * self.inverse_bins - 1
        fine_input = torch.cat([coarse_input, scaled_coarse_last], dim=0)
        # [3 * half_size, batch_size] + [3 * half_size, 2] * [2, batch_size] →
        # [3 * half_size, batch_size]
        fine_input_projection = torch.addmm(project_fine_bias, self.project_fine_input.weight,
                                            fine_input)
        fine_last_hidden = _gru_gates(fine_hidden_r, fine_hidden_i, fine_hidden_n,
                                      fine_input_projection, fine_last_hidden)

        # Predict
        # [half_size, batch_size] → [batch_size, bins]
        fine_bins = self.to_bins_fine(fine_last_hidden.transpose(0, 1))

        # [batch_size, bins] → [batch_size]
        if self.argmax:
            fine_last = fine_bins.max(dim=1)[1]
        else:
            fine_posterior = softmax(fine_bins, dim=1)
            fine_last = torch.multinomial(fine_posterior, 1).squeeze(1)

        return coarse_last, fine_last, coarse_last_hidden, fine_last_hidden

    @script_method
    def _infer_loop(self, project_hidden_weights, project_hidden_bias, project_coarse_bias,
                    project_fine_bias, coarse_last, fine_last, coarse_last_hidden,
                    fine_last_hidden):
        # [3 * half_size, signal_length, batch_size]
        _, signal_length, batch_size = project_coarse_bias.shape
        out_coarse = torch.zeros(batch_size, signal_length)
        out_fine = torch.zeros(batch_size, signal_length)
        for i in range(signal_length):
            coarse_last, fine_last, coarse_last_hidden, fine_last_hidden = self._infer_step_traced(
                coarse_last, coarse_last_hidden, fine_last, fine_last_hidden, project_hidden_bias,
                project_hidden_weights, project_coarse_bias[:, i], project_fine_bias[:, i])
            out_coarse[:, i] = coarse_last
            out_fine[:, i] = fine_last

        return out_coarse, out_fine, coarse_last_hidden, fine_last_hidden

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

    @log_runtime
    def forward(self, local_features, hidden_state=None, pad=True):
        """  Run WaveRNN in inference mode.

        Variables:
            r: ``r`` stands for reset gate
            i: ``i`` stands for input gate
            n: ``n`` stands for new gate

        Reference: # noqa
            * PyTorch GRU Equations
              https://pytorch.org/docs/stable/nn.html?highlight=gru#torch.nn.GRU
            * Efficient Neural Audio Synthesis Equations
              https://arxiv.org/abs/1802.08435
            * https://cs.stackexchange.com/questions/79241/what-is-temperature-in-lstm-and-neural-networks-generally

        Args:
            local_features (torch.FloatTensor [batch_size, local_length, local_features_size] or
                [local_length, local_features_size]): Local feature to condition signal generation
                (e.g. spectrogram).\
            hidden_state (tuple, optional): Initial hidden state with RNN hidden state and last
                coarse/fine samples.
            pad (bool, optional): Pad the spectrogram with zeros on the ends, assuming that the
                spectrogram has no context on the ends.

        Returns:
            out_coarse (torch.LongTensor [batch_size, signal_length] or [signal_length]): Predicted
                categorical distribution over ``bins`` categories for the ``coarse`` random
                variable.
            out_fine (torch.LongTensor [batch_size, signal_length] or [signal_length]): Predicted
                categorical distribution over ``bins`` categories for the ``fine`` random
                variable.
            hidden_state (tuple): Hidden state with RNN hidden state and last coarse/fine samples.
        """
        is_unbatched = len(local_features.shape) == 2
        if is_unbatched:
            # [local_length, local_features_size]  → [batch_size, local_length, local_features_size]
            local_features = local_features.unsqueeze(0)

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
        bias_r, bias_i, bias_n = self.stripped_gru.gru.bias_ih_l0.chunk(3)

        # ... [batch_size, signal_length, half_size]
        bias_coarse_r, bias_fine_r = torch.chunk(conditional[:, :, 0] + bias_r, 2, dim=2)
        bias_coarse_i, bias_fine_i = torch.chunk(conditional[:, :, 1] + bias_i, 2, dim=2)
        bias_coarse_n, bias_fine_n = torch.chunk(conditional[:, :, 2] + bias_n, 2, dim=2)

        # [batch_size, signal_length, half_size] → [batch_size, signal_length, 3 * half_size]
        bias_coarse = torch.cat((bias_coarse_r, bias_coarse_i, bias_coarse_n), dim=2)
        bias_fine = torch.cat((bias_fine_r, bias_fine_i, bias_fine_n), dim=2)

        del bias_coarse_r, bias_fine_r
        del bias_coarse_i, bias_fine_i
        del bias_coarse_n, bias_fine_n

        # [batch_size, signal_length, 3 * half_size] → [3 * half_size, signal_length, batch_size]
        project_coarse_bias = bias_coarse.transpose(0, 2)
        project_fine_bias = bias_fine.transpose(0, 2)

        project_hidden_weights = self._reorder_gru_weights(self.stripped_gru.gru.weight_hh_l0)
        project_hidden_bias = self._reorder_gru_weights(
            self.stripped_gru.gru.bias_hh_l0).unsqueeze(1)

        # Initial inputs
        coarse_last, fine_last, coarse_last_hidden, fine_last_hidden = self._infer_initial_state(
            conditional, batch_size, hidden_state)

        # Predict waveform
        out_coarse, out_fine, coarse_last_hidden, fine_last_hidden = log_runtime(self._infer_loop)(
            project_hidden_weights, project_hidden_bias, project_coarse_bias, project_fine_bias,
            coarse_last, fine_last, coarse_last_hidden, fine_last_hidden)

        hidden_state = (out_coarse[-1], out_fine[-1], coarse_last_hidden, fine_last_hidden)

        if is_unbatched:
            # NOTE: Hidden state remains batched.
            return out_coarse.squeeze(0), out_fine.squeeze(0), hidden_state

        return out_coarse, out_fine, hidden_state


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
    __constants__ = ['size', 'half_size', 'inverse_bins']

    @configurable
    def __init__(self,
                 hidden_size=ConfiguredArg(),
                 bits=ConfiguredArg(),
                 upsample_num_filters=ConfiguredArg(),
                 upsample_kernels=ConfiguredArg(),
                 upsample_repeat=ConfiguredArg(),
                 local_features_size=ConfiguredArg()):
        super(WaveRNN, self).__init__()

        assert hidden_size % 2 == 0, "Hidden size must be even."
        assert bits % 2 == 0, "Bits must be even for a double softmax"
        self.bits = bits
        self.bins = int(2**(bits / 2))  # Encode ``bits`` with double softmax of ``bits / 2``
        self.inverse_bins = 1 / ((self.bins - 1) / 2)
        self.size = hidden_size
        self.half_size = int(self.size / 2)

        # Output fully connected layers
        self.to_bins_coarse = nn.Sequential(
            nn.Linear(self.half_size, self.half_size), nn.ReLU(inplace=True),
            nn.Linear(self.half_size, self.bins))
        self.to_bins_fine = nn.Sequential(
            nn.Linear(self.half_size, self.half_size), nn.ReLU(inplace=True),
            nn.Linear(self.half_size, self.bins))

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

        self._init_weights()

    def _init_weights(self):
        """ Initialize the weight matricies for various layers """
        gain = torch.nn.init.calculate_gain('relu')
        torch.nn.init.orthogonal_(self.to_bins_fine[0].weight, gain=gain)
        torch.nn.init.orthogonal_(self.to_bins_coarse[0].weight, gain=gain)

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

    def to_inferrer(self, device=torch.device('cpu'), argmax=False):
        """ Create a inferrer ``torch.jit.ScriptModule`` on a particular device for inference.

        Args:
            device (torch.device, optional): Device this inference module is to run on.
            argmax (bool, optional): If ``True``, during sampling pick the most likely bin instead
                of sampling from a multinomial distribution.

        Returns:
            _WaveRNNInferrer
        """
        return _WaveRNNInferrer(
            hidden_size=self.size,
            bits=self.bits,
            to_bins_coarse=self.to_bins_coarse,
            to_bins_fine=self.to_bins_fine,
            project_coarse_input=self.project_coarse_input,
            project_fine_input=self.project_fine_input,
            conditional_features_upsample=self.conditional_features_upsample,
            stripped_gru=self.stripped_gru,
            argmax=argmax,
            device=device)

    def forward(self, local_features, input_signal, target_coarse, hidden_state=None, pad=False):
        """
        Note:
            - Forward does not support unbatched mode, yet unlike other model

        Args:
            local_features (torch.FloatTensor [batch_size, local_length, local_features_size]):
                Local feature to condition signal generation (e.g. spectrogram).
            input_signal (torch.FloatTensor [batch_size, signal_length, 2]): Course
                ``signal[:, :, 0]`` and fines values ``signal[:, :, 1]`` used for teacher forcing
                each between the range [-1, 1].
            target_coarse (torch.FloatTensor [batch_size, signal_length, 1]): Same as the
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
        input_signal = input_signal.float() * self.inverse_bins - 1
        target_coarse = target_coarse.float() * self.inverse_bins - 1

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

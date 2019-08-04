from pathlib import Path

import copy
import logging

from torch import nn
from torch.utils.cpp_extension import load_inline

import torch

from src.audio import split_signal
from src.environment import ROOT_PATH
from src.hparams import configurable
from src.hparams import ConfiguredArg
from src.signal_model.stripped_gru import StrippedGRU
from src.signal_model.upsample import ConditionalFeaturesUpsample
from src.utils import log_runtime

logger = logging.getLogger(__name__)


class _WaveRNNInferrer(nn.Module):
    """ `_WaveRNNInferrer` for computing fast inference on a CPU.

    Args:
        hidden_size (int): Provided by WaveRNN.
        to_bins_coarse (torch.nn.Module): Provided by WaveRNN.
        to_bins_fine (torch.nn.Module): Provided by WaveRNN.
        project_coarse_input (torch.nn.Module): Provided by WaveRNN.
        project_fine_input (torch.nn.Module): Provided by WaveRNN.
        conditional_features_upsample (torch.nn.Module): Provided by WaveRNN.
        stripped_gru (torch.nn.Module): Provided by WaveRNN.
        argmax (bool, optional): If ``True``, during sampling pick the most likely bin instead of
            sampling from a multinomial distribution.
    """

    @log_runtime
    def __init__(self,
                 hidden_size,
                 to_bins_coarse,
                 to_bins_fine,
                 project_coarse_input,
                 project_fine_input,
                 conditional_features_upsample,
                 stripped_gru,
                 argmax=False):
        super().__init__()

        logger.info('Instantiating `_WaveRNNInferrer`.')

        assert hidden_size % 64 == 0, 'Hidden size must be multiple of 64.'

        # See documentation for more information:
        # https: // pytorch.org/docs/stable/cpp_extension.html#torch.utils.cpp_extension.load
        assert torch.utils.cpp_extension.verify_ninja_availability(), (
            '`_WaveRNNInferrer` requires that `ninja` is installed, '
            'learn more here: via https://ninja-build.org/')

        cflags = ['-O2', '-funroll-loops', '-march=native']
        ldflags = []
        if Path('/usr/lib/x86_64-linux-gnu/mkl/').exists():
            # NOTE: The MKL path is hard coded assuming that this is running on a machine with
            # Ubuntu 18.10 or higher and has `libmkl-full-dev` installed.
            ldflags += ['-L/usr/lib/x86_64-linux-gnu/mkl/', '-lblas']
            cflags.append('-fopenmp')
        else:
            logger.warning(
                '`_WaveRNNInferrer` will be less performant because the requirements '
                'for MKL have not been met. For MKL, the program must be running on a Ubuntu 18.10 '
                'or higher machine with the `apt-get install libmkl-full-dev` package.')
            cflags.append('-DNO_MKL=1')

        logger.info('Building `inference.cpp` with `ninja`...')
        # NOTE: Learn more about `read_text` here:
        # https://github.com/wellsaid-labs/Benchmarks/pull/6/files#r306031034
        self._cpp_inference = load_inline(
            name='inference',
            cpp_sources=(Path(__file__).parent / 'inference.cpp').read_text(),
            extra_include_paths=[str(ROOT_PATH / 'third_party')],
            extra_cflags=cflags,
            extra_ldflags=ldflags,
            verbose=True)

        self.argmax = argmax
        self.size = hidden_size

        to_bins_coarse = to_bins_coarse.cpu()
        to_bins_fine = to_bins_fine.cpu()
        project_coarse_input = project_coarse_input.cpu()
        project_fine_input = project_fine_input.cpu()
        stripped_gru = stripped_gru.cpu()
        conditional_features_upsample = conditional_features_upsample.cpu()

        # Initialize to_bins_coarse.
        self.to_bins_coarse_pre_bias_aligned = to_bins_coarse[0].bias.clone()
        self.to_bins_coarse_pre_weight_t = to_bins_coarse[0].weight.t().clone()

        self.to_bins_coarse_bias_aligned = to_bins_coarse[2].bias.clone()
        self.to_bins_coarse_weight_t = to_bins_coarse[2].weight.t().clone()

        # Initialize to_bins_fine.
        self.to_bins_fine_pre_bias_aligned = to_bins_fine[0].bias.clone()
        self.to_bins_fine_pre_weight_t = to_bins_fine[0].weight.t().clone()

        self.to_bins_fine_bias_aligned = to_bins_fine[2].bias.clone()
        self.to_bins_fine_weight_t = to_bins_fine[2].weight.t().clone()

        # Initialize project_coarse_*
        self.project_coarse_input_weight_t = project_coarse_input.weight.t().clone()
        self.project_fine_input_weight_t = project_fine_input.weight.t().clone()

        self.project_hidden_bias_aligned = self._reorder_gru_weights(
            stripped_gru.gru.bias_hh_l0).clone()
        self.project_hidden_weight_aligned = self._reorder_gru_weights(
            stripped_gru.gru.weight_hh_l0).clone()

        # [self.size * 3] → bias_r, bias_u, bias_e [self.size]
        (self.stripped_gru_bias_r, self.stripped_gru_bias_i,
         self.stripped_gru_bias_n) = stripped_gru.gru.bias_ih_l0.clone().chunk(3)

        self.conditional_features_upsample = conditional_features_upsample

        self.cpu()  # Module only supports `cpu`
        self.eval()  # Evaluation mode by default

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

    def _infer_loop(self, coarse_last, fine_last, hidden_state, project_coarse_bias,
                    project_fine_bias):
        """ Run the inference loop of WaveRNN.

        Side effects:
            - `hidden_state` is modified in-place to be the updated hidden state
            - `project_coarse_bias` is modified in-place to some intermediate value.
            - `project_fine_bias` is modified in-place to some intermediate value.

        Args:
            coarse_last (torch.LongTensor [1])
            fine_last (torch.LongTensor [1])
            hidden_state (torch.FloatTensor [self.size])
            project_coarse_bias (torch.FloatTensor [sequence_length, 3 * half_size])
            project_fine_bias (torch.FloatTensor [sequence_length, 3 * half_size])

        Returns:
            out_coarse (torch.LongTensor [sequence_length]): Predicted categorical distribution over
              ``bins`` categories for the ``coarse`` random variable.
            out_fine (torch.LongTensor [sequence_length]): Predicted categorical distribution over
              ``bins`` categories for the ``fine`` random variable.
            hidden_state (torch.FloatTensor [self.size]): Hidden state with RNN hidden state.
        """
        with torch.no_grad():
            return tuple(
                self._cpp_inference.run(
                    coarse_last.clone(),
                    fine_last.clone(),
                    hidden_state.clone(),
                    project_coarse_bias.clone(),
                    self.project_coarse_input_weight_t,
                    project_fine_bias.clone(),
                    self.project_fine_input_weight_t,
                    self.project_hidden_bias_aligned,
                    self.project_hidden_weight_aligned,
                    self.to_bins_coarse_pre_bias_aligned,
                    self.to_bins_coarse_pre_weight_t,
                    self.to_bins_coarse_bias_aligned,
                    self.to_bins_coarse_weight_t,
                    self.to_bins_fine_pre_bias_aligned,
                    self.to_bins_fine_pre_weight_t,
                    self.to_bins_fine_bias_aligned,
                    self.to_bins_fine_weight_t,
                    self.argmax,
                ))

    def _infer_initial_state(self, reference, hidden_state=None):
        """ Initial state returns the initial hidden state and go sample.

        Args:
            reference (torch.Tensor): Tensor to reference device and dtype from.
            hidden_state (torch.FloatTensor [self.size], optional): GRU hidden state.

        Return:
            coarse (torch.LongTensor [1]): Initial coarse value from [0, 255]
            fine (torch.LongTensor [1]): Initial fine value from [0, 255]
            hidden_state (torch.FloatTensor [self.size]): Initial RNN hidden state.
        """
        if hidden_state is not None:
            assert len(hidden_state) == 3
            return hidden_state

        # Set the split signal value at signal value zero
        coarse, fine = split_signal(reference.new_zeros(1))
        hidden_state = reference.new_zeros(self.size)
        return coarse.long(), fine.long(), hidden_state

    @log_runtime
    def forward(self, local_features, hidden_state=None, pad=True):
        """  Run WaveRNN in inference mode.

        Variables:
            r: ``r`` stands for a reset gate component.
            i: ``i`` stands for a input gate component.
            n: ``n`` stands for a new gate component.

        Reference: # noqa
            * PyTorch GRU Equations
              https://pytorch.org/docs/stable/nn.html?highlight=gru#torch.nn.GRU
            * Efficient Neural Audio Synthesis Equations
              https://arxiv.org/abs/1802.08435
            * https://cs.stackexchange.com/questions/79241/what-is-temperature-in-lstm-and-neural-networks-generally

        Args:
            local_features (torch.FloatTensor [local_length, local_features_size]): Local feature to
                condition signal generation (e.g. spectrogram).
            hidden_state (tuple, optional): Initial hidden state with RNN hidden state and last
                coarse/fine samples.
            pad (bool, optional): Pad the spectrogram with zeros on the ends, assuming that the
                spectrogram has no context on the ends.

        Returns:
            out_coarse (torch.LongTensor [signal_length]): Predicted categorical distribution over
                ``bins`` categories for the ``coarse`` random variable.
            out_fine (torch.LongTensor [signal_length]): Predicted categorical distribution over
                ``bins`` categories for the ``fine`` random variable.
            hidden_state (tuple): Hidden state with RNN hidden state and last coarse/fine samples.
        """
        # [local_length, local_features_size] → [1, local_length, local_features_size]
        local_features = local_features.unsqueeze(0)

        # [1, local_length, local_features_size] →
        # [1, self.size * 3, signal_length]
        conditional = self.conditional_features_upsample(local_features, pad=pad)

        # [1, self.size * 3, signal_length] → [self.size * 3, signal_length]
        conditional = conditional.squeeze(0)

        _, signal_length = conditional.shape

        # [self.size * 3, signal_length] →
        # [signal_length, self.size * 3]
        conditional = conditional.transpose(0, 1)

        # [signal_length,  3 * self.size] →
        # [signal_length,  3, self.size]
        conditional = conditional.view(signal_length, 3, -1)

        # ... [signal_length, half_size]
        chunk = torch.chunk
        bias_coarse_r, bias_fine_r = chunk(conditional[:, 0] + self.stripped_gru_bias_r, 2, dim=1)
        bias_coarse_i, bias_fine_i = chunk(conditional[:, 1] + self.stripped_gru_bias_i, 2, dim=1)
        bias_coarse_n, bias_fine_n = chunk(conditional[:, 2] + self.stripped_gru_bias_n, 2, dim=1)

        # [signal_length, half_size] → [signal_length, 3 * half_size]
        bias_coarse = torch.cat((bias_coarse_r, bias_coarse_i, bias_coarse_n), dim=1)
        bias_fine = torch.cat((bias_fine_r, bias_fine_i, bias_fine_n), dim=1)

        # Initial inputs
        coarse_last, fine_last, gru_hidden_state = self._infer_initial_state(
            conditional, hidden_state)

        # Predict waveform
        out_coarse, out_fine, gru_hidden_state = log_runtime(self._infer_loop)(coarse_last,
                                                                               fine_last,
                                                                               gru_hidden_state,
                                                                               bias_coarse,
                                                                               bias_fine)

        return out_coarse, out_fine, (out_coarse[-1], out_fine[-1], gru_hidden_state)


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
    __constants__ = ['size', 'half_size', 'inverse_bins', 'bins']

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

    def to_inferrer(self, argmax=False):
        """ Instantiate the inferrer algorithm.

        Args:
            argmax (bool, optional): If ``True``, during sampling pick the most likely bin instead
                of sampling from a multinomial distribution.

        Returns:
            _WaveRNNInferrer
        """
        return _WaveRNNInferrer(
            hidden_size=self.size,
            to_bins_coarse=copy.deepcopy(self.to_bins_coarse),
            to_bins_fine=copy.deepcopy(self.to_bins_fine),
            project_coarse_input=copy.deepcopy(self.project_coarse_input),
            project_fine_input=copy.deepcopy(self.project_fine_input),
            conditional_features_upsample=copy.deepcopy(self.conditional_features_upsample),
            stripped_gru=copy.deepcopy(self.stripped_gru),
            argmax=argmax)

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

from torch import nn

import torch

from src.audio import split_signal
from src.utils import log_runtime
from src.hparams import configurable
from src.hparams import ConfiguredArg
from src.signal_model.stripped_gru import StrippedGRU
from src.signal_model.upsample import ConditionalFeaturesUpsample


def flatten(*args, **kwargs):
    """ Flatten tensors onto the same contiguous tensor.

    Args:
        *args: List of tensors to flatten.
        **kwargs: Keywords arguments passed to `torch.zeros` upon creating the flattened tensor.

    Returns:
        *args: List of tensors as views into a larger tensor.
    """
    total = sum([t.numel() for t in args])
    contiguous = torch.zeros(total, **kwargs)
    offset = 0
    return_ = []
    for tensor in args:
        num_elements = tensor.numel()
        view = contiguous[offset:num_elements + offset].view(tensor.shape)
        view.copy_(tensor)
        return_.append(view)
        offset += num_elements
    return tuple(return_)


class _WaveRNNInferrer(nn.Module):
    """ WaveRNN JIT for computing inference.

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
    __constants__ = ['size', 'half_size', 'inverse_bins', 'bins']

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

    def _infer_half_step(self, input_, last_hidden, input_bias, input_weight, input_projection_out,
                         hidden_projection, to_bins_pre_bias, to_bins_pre_weight, to_bins_bias,
                         to_bins_weight, pre_bins_out, bins_out):
        """
        Side effects:
            - `last_hidden` is modified in-place to be the updated hidden state
            - `input_bias` is modified in-place to some intermediate value.
            - `pre_bins_out` is modified in-place to some intermediate value.
            - `bins_out` is modified in-place to some intermediate value.

        Args:
            input_ (torch.FloatTensor [n])
            last_hidden (torch.FloatTensor [half_size])
            input_bias (torch.FloatTensor [3 * half_size])
            input_weight (torch.FloatTensor [3 * half_size, n])
            hidden_projection (torch.FloatTensor [3 * half_size])
            pre_bins_out (torch.FloatTensor [half_size]): Memory space to store intermediate value.
            bins_out (torch.FloatTensor [num_bins]): Memory space to store intermediate value.
            ...

        Returns:
            predicted (torch.LongTensor [1])
        """
        # [3 * half_size] + [3 * half_size, n] * [n] → [3 * half_size]
        torch.addmv(input_bias, input_weight, input_, out=input_projection_out)

        # Compute GRU gates [half_size]

        # Compute `torch.sigmoid(input_ri + hidden_ri)`, consuming `input_projection[:self.size]]`
        input_ri = input_projection_out[:self.size]
        input_ri.add_(hidden_projection[:self.size])
        input_ri.sigmoid_()
        input_n = input_projection_out[self.size:]

        # Compute `new_gate = torch.tanh(input_n + reset_gate * hidden_n)`, consuming `input_n`
        input_n.addcmul_(1, input_ri[:self.half_size], hidden_projection[self.size:])
        input_n.tanh_()
        new_gate = input_n

        # Compute `last_hidden = new_gate + input_gate * (last_hidden - new_gate)`, consuming
        # `last_hidden`
        last_hidden.sub_(new_gate)
        last_hidden.mul_(input_ri[self.half_size:])
        last_hidden.add_(new_gate)

        # [half_size] + [half_size, half_size] * [half_size] → [half_size]
        torch.addmv(to_bins_pre_bias, to_bins_pre_weight, last_hidden, out=pre_bins_out)
        pre_bins_out.relu_()

        # [num_bins] + [num_bins, half_size] * [half_size] → [num_bins]
        torch.addmv(to_bins_bias, to_bins_weight, pre_bins_out, out=bins_out)

        # [num_bins] → [1]
        if self.argmax:
            return bins_out.max(dim=0)[1]

        posterior = torch.nn.functional.softmax(bins_out, dim=0)
        return torch.multinomial(posterior, 1).squeeze(0)

    def _infer_step(self, hidden_state, project_hidden_bias, project_hidden_weight,
                    project_coarse_bias, project_coarse_weight, project_fine_bias,
                    project_fine_weight, to_bins_coarse_pre_bias, to_bins_coarse_pre_weight,
                    to_bins_coarse_bias, to_bins_coarse_weight, to_bins_fine_pre_bias,
                    to_bins_fine_pre_weight, to_bins_fine_bias, to_bins_fine_weight, input_out,
                    input_projection_out, hidden_projection_out, pre_bins_out, bins_out):
        """
        Side effects:
            - `last_hidden` is modified in-place to be the updated hidden state
            - `project_coarse_bias` is modified in-place to some intermediate value.
            - `project_fine_bias` is modified in-place to some intermediate value.
            - `pre_bins_out` is modified in-place to some intermediate value.
            - `bins_out` is modified in-place to some intermediate value.
            - `hidden_projection_out` is modified in-place to some intermediate value.
            - `input_out` is modified in-place to some intermediate value.

        Args:
            hidden_state (torch.FloatTensor [size])
            project_hidden_bias (torch.FloatTensor [3 * size]): Project hidden bias should be
                ordered like so `[coarse r, coarse i, coarse n, fine_r, fine_i, fine_n,]`
            project_hidden_weight (torch.FloatTensor [3 * size, size]): Ditto.
            project_coarse_bias (torch.FloatTensor [3 * half_size])
            project_fine_bias (torch.FloatTensor [3 * half_size])
            input_out (torch.FloatTensor [3]): Memory space to store intermediate values. Also
                `input_out` must be initialized such that
                 `input_out[0:2] = [coarse bit, fine bit]` scaled between [-1, 1].
            hidden_projection_out (torch.FloatTensor [3 * size]): Memory space to store intermediate
                value.
            pre_bins_out (torch.FloatTensor [half_size]): Pre-allocated memory space to store
                intermediate value.
            bins_out (torch.FloatTensor [num_bins]): Pre-allocated memory space to store
                intermediate value.
            ...

        Returns:
            coarse_last (torch.LongTensor [1])
            fine_last (torch.LongTensor [1])
        """
        # [3 * size] + [3 * size, size] * [size] = [3 * size]
        torch.addmv(
            project_hidden_bias, project_hidden_weight, hidden_state, out=hidden_projection_out)

        coarse_last = self._infer_half_step(
            input_=input_out[:2],
            last_hidden=hidden_state[:self.half_size],
            input_bias=project_coarse_bias,
            input_weight=project_coarse_weight,
            hidden_projection=hidden_projection_out[:self.half_size * 3],
            to_bins_pre_bias=to_bins_coarse_pre_bias,
            to_bins_pre_weight=to_bins_coarse_pre_weight,
            to_bins_bias=to_bins_coarse_bias,
            to_bins_weight=to_bins_coarse_weight,
            input_projection_out=input_projection_out,
            pre_bins_out=pre_bins_out,
            bins_out=bins_out)
        return_coarse_last = coarse_last.clone()
        input_out[2] = coarse_last
        input_out[2].mul_(self.inverse_bins)
        input_out[2].sub_(1)

        fine_last = self._infer_half_step(
            input_=input_out,
            last_hidden=hidden_state[self.half_size:],
            input_bias=project_fine_bias,
            input_weight=project_fine_weight,
            hidden_projection=hidden_projection_out[self.half_size * 3:],
            to_bins_pre_bias=to_bins_fine_pre_bias,
            to_bins_pre_weight=to_bins_fine_pre_weight,
            to_bins_bias=to_bins_fine_bias,
            to_bins_weight=to_bins_fine_weight,
            input_projection_out=input_projection_out,
            pre_bins_out=pre_bins_out,
            bins_out=bins_out)
        return_fine_last = fine_last.clone()
        input_out[1] = fine_last
        input_out[1].mul_(self.inverse_bins)
        input_out[1].sub_(1)

        input_out[0] = input_out[2]  # Move `coarse t` to `coarse t - 1`

        return return_coarse_last, return_fine_last

    def _infer_loop(self, project_hidden_weight, project_hidden_bias, project_coarse_bias,
                    project_fine_bias, coarse_last, fine_last, hidden_state, input_out):
        """
        Side effects:
            - `hidden_state` is modified in-place to be the updated hidden state
            - `project_coarse_bias` is modified in-place to some intermediate value.
            - `project_fine_bias` is modified in-place to some intermediate value.

        Args:
            project_hidden_weight (torch.FloatTensor [3 * size, size])
            project_hidden_bias (torch.FloatTensor [3 * size])
            project_coarse_bias (torch.FloatTensor [sequence_length, 3 * half_size])
            project_fine_bias (torch.FloatTensor [sequence_length, 3 * half_size])
            coarse_last (torch.LongTensor [1])
            fine_last (torch.LongTensor [1])
            hidden_state (torch.FloatTensor [size])
            input_out (torch.FloatTensor [3])

        Returns:
            out_coarse (torch.LongTensor [signal_length])
            out_fine (torch.LongTensor [signal_length])
            hidden_sate (torch.FloatTensor [size])
        """
        # [signal_length, 3 * half_size]
        device = project_hidden_weight.device
        signal_length, _ = project_coarse_bias.shape

        out_coarse = torch.zeros(signal_length, dtype=torch.long, device=device)
        out_fine = torch.zeros(signal_length, dtype=torch.long, device=device)

        # Setup memory for the output
        pre_bins_out = torch.zeros(self.half_size, device=device)
        bins_out = torch.zeros(self.bins, device=device)
        hidden_projection_out = torch.zeros(3 * self.size, device=device)
        input_projection_out = torch.zeros(3 * self.half_size, device=device)

        # 15% speed up ensuring that data is contiguous
        project_hidden_bias, project_hidden_weight, hidden_state, hidden_projection_out = flatten(
            project_hidden_bias,
            project_hidden_weight,
            hidden_state,
            hidden_projection_out,
            device=device)

        (project_coarse_weight, input_out, input_projection_out, to_bins_coarse_pre_bias,
         to_bins_coarse_pre_weight, pre_bins_out, to_bins_coarse_bias, to_bins_coarse_weight,
         bins_out, project_fine_weight, to_bins_fine_pre_bias, to_bins_fine_pre_weight,
         to_bins_fine_bias, to_bins_fine_weight) = flatten(
             self.project_coarse_input.weight,
             input_out,
             input_projection_out,
             self.to_bins_coarse[0].bias,
             self.to_bins_coarse[0].weight,
             pre_bins_out,
             self.to_bins_coarse[2].bias,
             self.to_bins_coarse[2].weight,
             bins_out,
             self.project_fine_input.weight,
             self.to_bins_fine[0].bias,
             self.to_bins_fine[0].weight,
             self.to_bins_fine[2].bias,
             self.to_bins_fine[2].weight,
             device=device)

        for i in range(signal_length):
            coarse_last, fine_last = self._infer_step(
                hidden_state=hidden_state,
                project_hidden_bias=project_hidden_bias,
                project_hidden_weight=project_hidden_weight,
                project_coarse_bias=project_coarse_bias[i],
                project_coarse_weight=project_coarse_weight,
                project_fine_bias=project_fine_bias[i],
                project_fine_weight=project_fine_weight,
                to_bins_coarse_pre_bias=to_bins_coarse_pre_bias,
                to_bins_coarse_pre_weight=to_bins_coarse_pre_weight,
                to_bins_coarse_bias=to_bins_coarse_bias,
                to_bins_coarse_weight=to_bins_coarse_weight,
                to_bins_fine_pre_bias=to_bins_fine_pre_bias,
                to_bins_fine_pre_weight=to_bins_fine_pre_weight,
                to_bins_fine_bias=to_bins_fine_bias,
                to_bins_fine_weight=to_bins_fine_weight,
                input_out=input_out,
                input_projection_out=input_projection_out,
                hidden_projection_out=hidden_projection_out,
                pre_bins_out=pre_bins_out,
                bins_out=bins_out)

            out_coarse[i] = coarse_last
            out_fine[i] = fine_last

        return out_coarse, out_fine, hidden_state

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
        device = local_features.device

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
        conditional = conditional.view(signal_length, 3, self.size)

        # [size * 3] → bias_r, bias_u, bias_e [size]
        bias_r, bias_i, bias_n = self.stripped_gru.gru.bias_ih_l0.chunk(3)

        # ... [signal_length, half_size]
        bias_coarse_r, bias_fine_r = torch.chunk(conditional[:, 0] + bias_r, 2, dim=1)
        bias_coarse_i, bias_fine_i = torch.chunk(conditional[:, 1] + bias_i, 2, dim=1)
        bias_coarse_n, bias_fine_n = torch.chunk(conditional[:, 2] + bias_n, 2, dim=1)

        # [signal_length, half_size] → [signal_length, 3 * half_size]
        bias_coarse = torch.cat((bias_coarse_r, bias_coarse_i, bias_coarse_n), dim=1)
        bias_fine = torch.cat((bias_fine_r, bias_fine_i, bias_fine_n), dim=1)

        del bias_coarse_r, bias_fine_r
        del bias_coarse_i, bias_fine_i
        del bias_coarse_n, bias_fine_n

        project_hidden_weight = self._reorder_gru_weights(self.stripped_gru.gru.weight_hh_l0)
        project_hidden_bias = self._reorder_gru_weights(self.stripped_gru.gru.bias_hh_l0)

        # Initial inputs
        coarse_last, fine_last, gru_hidden_state = self._infer_initial_state(
            conditional, hidden_state)

        # Create memory for coarse / fine inputs
        input_out = torch.zeros(3, device=device)
        input_out[0] = coarse_last
        input_out[1] = fine_last
        input_out *= self.inverse_bins
        input_out -= 1

        # Predict waveform
        out_coarse, out_fine, hidden_state = log_runtime(
            self._infer_loop)(project_hidden_weight, project_hidden_bias, bias_coarse, bias_fine,
                              coarse_last, fine_last, gru_hidden_state, input_out)

        hidden_state = (out_coarse[-1], out_fine[-1], hidden_state)
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

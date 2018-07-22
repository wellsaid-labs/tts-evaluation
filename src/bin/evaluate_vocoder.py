import argparse
import logging
import os

from torch.nn.functional import log_softmax
from tqdm import tqdm

import librosa
import torch

from src.bin.signal_model._data_iterator import RandomSampler
from src.bin.signal_model._utils import load_checkpoint
from src.bin.signal_model._utils import load_data
from src.bin.signal_model._utils import set_hparams
from src.utils import combine_signal
from src.utils import split_signal
from src.utils.configurable import configurable

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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


def infer(self, local_features, argmax=False, temperature=1.0, hidden_state=None):
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
    # [batch_size, 3, self.size, signal_length]
    conditional = self.conditional_features_upsample(local_features)

    # [batch_size, 3, self.size, signal_length] →
    # [batch_size, signal_length, 3, self.size]
    conditional = conditional.permute(0, 3, 1, 2)

    batch_size, signal_length, _, _ = conditional.shape

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

    project_hidden_weights = _reorder_gru_weights(self, self.stripped_gru.gru.weight_hh_l0)
    project_hidden_bias = _reorder_gru_weights(self, self.stripped_gru.gru.bias_hh_l0).unsqueeze(1)

    # Initial inputs
    out_coarse, out_fine = [], []
    coarse, fine, coarse_last_hidden, fine_last_hidden = _infer_initial_state(
        self, conditional, batch_size, hidden_state)
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
        coarse_last_hidden, coarse = _infer_step(
            self,
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
        fine_last_hidden, fine = _infer_step(
            self,
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


@configurable
def main(checkpoint_path,
         results_path='results/',
         sample_rate=24000,
         samples=25,
         device=torch.device('cpu')):
    """ Generate random samples of vocoder to evaluate.

    Args:
        checkpoint_path (str): Checkpoint to load.
        results_path (str): Path to store results.
        sample_rate (int): Sample rate of audio evaluated.
        samples (int): Number of rows to evaluate.
        device (torch.device): Device on which to evaluate on.
    """
    os.makedirs(results_path, exist_ok=True)

    assert os.path.isfile(checkpoint_path)
    checkpoint = load_checkpoint(checkpoint_path, device=device)
    logger.info('Loaded checkpoint: %s', checkpoint)

    train, dev = load_data()

    torch.set_grad_enabled(False)
    model = checkpoint['model'].eval().to(device)

    for i, j in enumerate(RandomSampler(dev)):
        if i > samples:
            break

        logger.info('Evaluating dev row %d [%d of %d]', j, i + 1, samples)
        row = dev[j]

        # [batch_size, local_length, local_features_size]
        log_mel_spectrogram = row['log_mel_spectrogram'].unsqueeze(0).to(device)

        # [signal_length]
        signal = row['signal'].numpy()
        gold_path = os.path.join(results_path, 'gold_%d.wav' % j)
        librosa.output.write_wav(gold_path, signal, sr=sample_rate)
        logger.info('Saved file %s', gold_path)

        predicted_coarse, predicted_fine, _ = infer(model, log_mel_spectrogram)
        predicted_signal = combine_signal(predicted_coarse.squeeze(0),
                                          predicted_fine.squeeze(0)).numpy()

        predicted_path = os.path.join(results_path, 'predicted_%d.wav' % j)
        librosa.output.write_wav(predicted_path, predicted_signal, sr=sample_rate)
        logger.info('Saved file %s', predicted_path)
        print('-' * 100)


if __name__ == '__main__':  # pragma: no cover
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c', '--checkpoint', type=str, required=True, help='Vocoder checkpoint to evaluate.')
    cli_args = parser.parse_args()

    set_hparams()
    main(checkpoint_path=cli_args.checkpoint)

import statistics
import time
import logging

from tqdm import tqdm

# NOTE: Needs to be imported before torch
# Remove after this issue is resolved https://github.com/comet-ml/issue-tracking/issues/178
import comet_ml  # noqa

from torch.jit import script_method
from torch.jit import ScriptModule
from torch.jit import trace

import torch
import numpy

from src.audio import get_log_mel_spectrogram
from src.bin.train.signal_model.data_loader import _get_slice
from src.hparams import set_hparams
from src.signal_model import WaveRNN
from src.spectrogram_model.attention import LocationSensitiveAttention
from src.spectrogram_model.encoder import Encoder
from src.utils import collate_sequences
from src.utils import tensors_to

logger = logging.getLogger(__name__)


def _benchmark(args, callable_, device, kwargs={}, num_runs=20):
    """ Benchmark the ``callable_`` given ``args`` and ``kwargs`` as input on ``device``.
    """
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    # Warm up, first run is typically very slow
    callable_(*args, **kwargs)

    # Run model callable
    times = []
    for _ in tqdm(range(num_runs)):
        # Use ``synchronize`` to ensure some async CUDA operations are finished
        if device.type == 'cuda':
            torch.cuda.synchronize()
        start = time.time()
        callable_(*args, **kwargs)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        end = time.time()
        times.append((end - start) * 1000)  # From seconds to milliseconds

    # Compute benchmark
    logger.info('Median: %s milliseconds', statistics.median(times))
    logger.info('Mean: %s milliseconds', statistics.mean(times))
    if len(times) > 1:
        logger.info('Standard Deviation: %s milliseconds', statistics.stdev(times))
    logger.info('=' * 100)


def wave_rnn_batch_size(batch_size=64,
                        spectrogram_frame_channels=128,
                        samples=157680,
                        num_runs=10,
                        device=None):
    """ Benchmark the effect of batch size on the signal model run time.

    Learnings:
        - There is a cliff at ``batch_size=32`` where the performance decreases substantially.
        - 90% of the model time is spent computing the ``nn.GRU``, it'd be interesting to use
          another component like SRU instead.
        - ``torch.backends.cudnn.benchmark`` decreases the runtime a bit (3 of 500 milliseconds)
        - Test a 16-bit model too, NVIDIA has increased throughput for that setting.
        - Test sparsification of 95%, it may help training time performance.

    Args:
        spectrogram_frame_channels (int): Number of frame channels
        samples (int): Number of signal samples.
        num_runs (int): Number of runs of the function to benchmark.
        device (torch.device or None)
    """
    for batch_size in range(1, 64):
        logger.info('Benchmarking Signal Model with batch size: %d', batch_size)

        # Make inputs
        slices = []
        for _ in range(batch_size):
            signal = torch.empty(samples).uniform_(-1.0, 1.0)
            spectrogram, padding = get_log_mel_spectrogram(signal)
            signal = numpy.pad(signal, padding, mode='constant', constant_values=0)
            signal = torch.from_numpy(signal)
            spectrogram = torch.from_numpy(spectrogram)
            slices.append(_get_slice(spectrogram, signal))

        batch = collate_sequences(slices)
        batch = tensors_to(batch, device=device)
        net = WaveRNN(local_features_size=spectrogram_frame_channels).to(device)

        def run(net, batch):
            predicted_coarse, predicted_fine, _ = net.forward(
                local_features=batch.input_spectrogram[0],
                input_signal=batch.input_signal[0],
                target_coarse=batch.target_signal_coarse[0].unsqueeze(2))
            (predicted_coarse + predicted_fine).sum().backward()

        _benchmark((net, batch), run, device=device)


def encoder(batch_size=64, num_tokens=60, vocab_size=90, num_speakers=10, device=None):
    """ Bench mark the ``Encoder`` layer. """
    # Make inputs
    tokens = torch.LongTensor(batch_size, num_tokens).random_(1, vocab_size).to(device)
    speaker = torch.LongTensor(batch_size).random_(1, num_speakers).to(device)

    # Make Network
    net = Encoder(vocab_size=vocab_size, num_speakers=num_speakers).to(device)
    jit_net = trace(net, (tokens, speaker))

    logger.info('Benchmarking Encoder...')
    _benchmark((tokens, speaker), lambda t, s: net(t, s).sum().backward(), device=device)

    logger.info('Benchmarking Encoder compiled by JIT...')
    _benchmark((tokens, speaker), lambda t, s: jit_net(t, s).sum().backward(), device=device)


class _JITAttention(ScriptModule):
    """ JIT Attention module to bench mark with the ``align`` function. """

    def __init__(self, align):
        super().__init__()
        self.align = align

    @script_method
    def forward(self, context, encoded_tokens, tokens_mask, query, cumulative_alignment,
                iterations):
        # type: (Tensor, Tensor, Tensor, Tensor, Tensor, int)
        for _ in range(iterations):
            context, cumulative_alignment, _ = self.align(
                encoded_tokens, tokens_mask, query, cumulative_alignment=cumulative_alignment)
        return context.sum()


class _Attention(torch.nn.Module):
    """ Attention module to bench mark with the ``align`` function. """

    def __init__(self, align):
        super().__init__()
        self.align = align

    def forward(self, context, encoded_tokens, tokens_mask, query, cumulative_alignment,
                iterations):
        for _ in range(iterations):
            context, cumulative_alignment, _ = self.align(
                encoded_tokens, tokens_mask, query, cumulative_alignment=cumulative_alignment)
        return context.sum()


def attention(batch_size=64,
              num_tokens=60,
              attention_hidden_size=128,
              query_hidden_size=128,
              device=None):
    """ Bench mark the ``Attention`` layer. """
    encoded_tokens = torch.rand(num_tokens, batch_size, attention_hidden_size).to(device)
    tokens_mask = torch.zeros(batch_size, num_tokens).byte().to(device)
    query = torch.rand(1, batch_size, query_hidden_size).to(device)
    cumulative_alignment = torch.rand(batch_size, num_tokens).to(device)
    context = torch.empty(batch_size, attention_hidden_size)
    iterations = 100
    input_ = (context, encoded_tokens, tokens_mask, query, cumulative_alignment, iterations)
    attention = LocationSensitiveAttention(
        hidden_size=attention_hidden_size, query_hidden_size=query_hidden_size).to(device)

    logger.info('Benchmarking Attention...')
    _attention = _Attention(attention)
    _benchmark(input_, lambda *args: _attention(*args).backward(), device=device)

    logger.info('Benchmarking Attention compiled by JIT...')
    jit_attention = _JITAttention(
        trace(attention, (encoded_tokens, tokens_mask, query, cumulative_alignment)))
    _benchmark(input_, lambda *args: jit_attention(*args).backward(), device=device)


def wave_rnn_inference(batch_size=1, spectrogram_length=80, spectrogram_channels=128, device=None):
    """ Bench mark the ``Attention`` layer. """
    spectrogram = torch.rand(batch_size, spectrogram_length, spectrogram_channels).to(device)
    wave_rnn = WaveRNN(device=device).eval()

    logger.info('Benchmarking WaveRNN...')
    with torch.no_grad():
        _benchmark((spectrogram,), wave_rnn.infer, device=device, num_runs=1)


if __name__ == "__main__":
    set_hparams()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # encoder(device=device)
    # attention(device=device)
    wave_rnn_inference(device=device)

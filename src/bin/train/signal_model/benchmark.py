import statistics
import time
import logging

from tqdm import tqdm

# NOTE: Needs to be imported before torch
# Remove after this issue is resolved https://github.com/comet-ml/issue-tracking/issues/178
import comet_ml  # noqa

import numpy
import torch

from src.audio import get_log_mel_spectrogram
from src.bin.train.signal_model.data_loader import _get_slice
from src.hparams import configurable
from src.hparams import set_hparams
from src.signal_model import WaveRNN
from src.utils import collate_sequences
from src.utils import tensors_to

logger = logging.getLogger(__name__)

# TODO: Move this to a notebook, it's more appropriate to demonstrate findings.


@configurable
def signal_model_batch_size(batch_size,
                            spectrogram_frame_channels,
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

    if device is None:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    for batch_size in range(1, 64):
        logger.info('Batch size: %d', batch_size)

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
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

        # Run model
        times = []
        for _ in tqdm(range(num_runs)):
            # Use ``synchronize`` to ensure some async CUDA operations are finished
            torch.cuda.synchronize()
            start = time.time()
            predicted_coarse, predicted_fine, _ = net.forward(
                local_features=batch.input_spectrogram[0],
                input_signal=batch.input_signal[0],
                target_coarse=batch.target_signal_coarse[0].unsqueeze(2))
            (predicted_coarse + predicted_fine).sum().backward()
            torch.cuda.synchronize()
            end = time.time()
            times.append((end - start) * 1000)  # From seconds to milliseconds

        # Compute benchmark
        logger.info('Median: %s milliseconds', statistics.median(times))
        logger.info('Mean: %s milliseconds', statistics.mean(times))
        logger.info('Standard Deviation: %s milliseconds', statistics.stdev(times))


if __name__ == "__main__":
    set_hparams()
    signal_model_batch_size()

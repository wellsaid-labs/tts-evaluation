import logging
import random

from torch.utils import data

import torch
import numpy as np

from src.utils.configurable import configurable
from src.utils import get_filename_table
from src.utils import split_signal
from src.utils import combine_signal

logger = logging.getLogger(__name__)


class SignalDataset(data.Dataset):
    """ Signal dataset loads and preprocesses a spectrogram and signal for training.

    Args:
        source (str): Directory with data.
        log_mel_spectrogram_prefix (str): Prefix of log mel spectrogram files.
        signal_prefix (str): Prefix of signal files.
        extension (str): Filename extension to load.
        frame_size (int, optional): Frame size to sample.
        random (random.Random, optional): Random number generator to sample data.

    References:
        * Parallel WaveNet https://arxiv.org/pdf/1711.10433.pdf
          "each containing 7,680 timesteps (roughly 320ms)."
    """

    @configurable
    def __init__(self,
                 source,
                 log_mel_spectrogram_prefix='log_mel_spectrogram',
                 signal_prefix='signal',
                 extension='.npy',
                 frame_size=3,
                 random=random):
        prefixes = [log_mel_spectrogram_prefix, signal_prefix]
        self.rows = get_filename_table(source, prefixes=prefixes, extension=extension)
        self.log_mel_spectrogram_prefix = log_mel_spectrogram_prefix
        self.signal_prefix = signal_prefix
        self.frame_size = frame_size
        self.random = random

    def __len__(self):
        return len(self.rows)

    def _get_slice(self, log_mel_spectrogram, signal):
        """ Slice the data into bite sized chunks that fit onto GPU memory for training.

        Notes:
            * Frames batch needs to line up with the target signal. Each frame, is used to predict
              the target. While for the source singal, we use the last output to predict the
              target; therefore, the source signal is one timestep behind.
            * Source signal batch is one time step behind the target batch. They have the same
              signal lengths.

        Args:
            log_mel_spectrogram (torch.Tensor [num_frames, channels])
            signal (torch.Tensor [signal_length])

        Returns:
            input_signal (torch.Tensor [signal_length, 2])
            frames_slice (torch.Tensor [num_frames, channels])
            target_signal_coarse (torch.Tensor [signal_length])
            target_signal_fine (torch.Tensor [signal_length])
        """
        samples, num_frames = signal.shape[0], log_mel_spectrogram.shape[0]
        samples_per_frame = int(samples / num_frames)

        # Signal model requires that there is a scaling factor between the signal and frames
        assert samples % num_frames == 0

        # Get a frame slice
        # ``-frame_size + 1, num_frames - 1`` to ensure there is an equal chance to that a
        # sample will be included inside the slice.
        # For example, with signal ``[1, 2, 3]`` and a ``slice_samples`` of 2 you'd get slices of:
        # (1), (1, 2), (2, 3), (3).
        # With each number represented at twice.
        start_frame = max(self.random.randint(-self.frame_size + 1, num_frames - 1), 0)
        end_frame = min(start_frame + self.frame_size, num_frames)
        frames_slice = log_mel_spectrogram[start_frame:end_frame]

        # Get a source sample slice shifted back one and target sample
        end_sample = end_frame * samples_per_frame
        start_sample = start_frame * samples_per_frame
        source_signal_slice = signal[max(start_sample - 1, 0):end_sample - 1]
        target_signal_slice = signal[start_sample:end_sample]

        # EDGE CASE: Add a go sample for source
        if start_sample == 0:
            go_sample = signal.new_zeros(1)
            source_signal_slice = torch.cat((go_sample, source_signal_slice), dim=0)

        source_signal_coarse, source_signal_fine = split_signal(source_signal_slice)
        target_signal_coarse, target_signal_fine = split_signal(target_signal_slice)

        input_signal = torch.stack((source_signal_coarse, source_signal_fine), dim=1)

        return {
            'input_signal': input_signal,
            'log_mel_spectrogram': frames_slice,
            'target_signal_coarse': target_signal_coarse,
            'target_signal_fine': target_signal_fine
        }

    def __getitem__(self, index):
        # signal [signal_length]
        signal = torch.from_numpy(np.load(self.rows[index][self.signal_prefix])).contiguous()
        signal = combine_signal(*split_signal(signal))  # Introduce quantization noise

        # log_mel_spectrogram [num_frames, channels]
        log_mel_spectrogram = torch.from_numpy(
            np.load(self.rows[index][self.log_mel_spectrogram_prefix])).contiguous()

        slice_ = self._get_slice(log_mel_spectrogram, signal)

        return {'slice': slice_, 'log_mel_spectrogram': log_mel_spectrogram, 'signal': signal}

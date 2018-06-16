import math
import random

from torch.utils import data
from torch import nn

import torch
import numpy as np

from src.audio import mu_law
from src.audio import mu_law_decode
from src.audio import mu_law_encode
from src.utils.configurable import configurable
from src.utils import get_filename_table


class SignalDataset(data.Dataset):
    """ Signal dataset loads and preprocesses a spectrogram and signal for training.

    Args:
        source (str): Directory with data.
        log_mel_spectrogram_prefix (str): Prefix of log mel spectrogram files.
        signal_prefix (str): Prefix of signal files.
        extension (str): Filename extension to load.
        slice_size (int): Size of slices to load for training data.
        receptive_field_size (int): Context added to slice; to compute target signal.

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
                 slice_size=7000,
                 receptive_field_size=1):
        # Invariant: Must be at least one. ``receptive_field_size`` includes the current timestep
        # that must be taken into consideration at very least to predict the next timestep.
        assert receptive_field_size >= 1
        prefixes = [log_mel_spectrogram_prefix, signal_prefix]
        self.rows = get_filename_table(source, prefixes=prefixes, extension=extension)
        self.slice_samples = slice_size
        self.set_receptive_field_size(receptive_field_size)
        self.log_mel_spectrogram_prefix = log_mel_spectrogram_prefix
        self.signal_prefix = signal_prefix

    def __len__(self):
        return len(self.rows)

    def set_receptive_field_size(self, receptive_field_size):
        """
        Args:
            receptive_field_size (int): Context added to slice; to compute target signal.
        """
        # Remove one, because the current sample is not tallied as context
        self.context_samples = receptive_field_size - 1

    def _preprocess(self, log_mel_spectrogram, signal):
        """ Slice the data into bite sized chunks that fit onto GPU memory for training.

        Notes:
            * Frames batch needs to line up with the target signal. Each frame, is used to predict
              the target. While for the source singal, we use the last output to predict the
              target; therefore, the source signal is one timestep behind.
            * Source signal batch is one time step behind the target batch. They have the same
              signal lengths.
            * With a large batch size and 1s+ clips, its probable that every batch will have at
              least one sample with full context; therefore, rather than aligning source signal
              with the target signal later adding more computation we align them now with left
              padding now by ensuring the context size is the same.
            * Following this comment from WaveNet authors:
              https://github.com/ibab/tensorflow-wavenet/issues/47#issuecomment-249080343
              We only encode the target signal and not the source signal.

        Args:
            log_mel_spectrogram (torch.Tensor [num_frames, channels])
            signal (torch.Tensor [signal_length])

        Returns:
            (dict): Dictionary with slices up to ``max_samples`` appropriate size for training.
        """
        samples, num_frames = signal.shape[0], log_mel_spectrogram.shape[0]
        samples_per_frame = int(samples / num_frames)
        slice_frames = int(self.slice_samples / samples_per_frame)
        context_frames = int(math.ceil(self.context_samples / samples_per_frame))

        # Invariants
        assert self.slice_samples % samples_per_frame == 0
        # Signal model requires that there is a scaling factor between the signal and frames
        assert samples % num_frames == 0

        # Get a frame slice
        # ``-slice_frames + 1, num_frames - 1`` to ensure there is an equal chance to that a
        # sample will be included inside the slice.
        # For example, with signal ``[1, 2, 3]`` and a ``slice_samples`` of 2 you'd get slices of:
        # (1), (1, 2), (2, 3), (3).
        # With each number represented at twice.
        start_frame = max(random.randint(-slice_frames + 1, num_frames - 1), 0)
        end_frame = min(start_frame + slice_frames, num_frames)
        start_context_frame = max(start_frame - context_frames, 0)
        frames_slice = log_mel_spectrogram[start_context_frame:end_frame]

        # Get a source sample slice shifted back one and target sample
        start_context_sample = start_context_frame * samples_per_frame
        end_sample = end_frame * samples_per_frame
        start_sample = start_frame * samples_per_frame
        source_signal_slice = signal[max(start_context_sample - 1, 0):end_sample - 1]
        target_signal_slice = mu_law_encode(signal[start_sample:end_sample])

        # EDGE CASE: Pad context incase it's cut off and add a go sample for source
        if start_context_frame == 0:
            go_sample = signal.new_zeros(1)
            source_signal_slice = torch.cat((go_sample, source_signal_slice), dim=0)

            context_frame_pad = context_frames - start_frame
            frames_slice = nn.functional.pad(frames_slice, (0, 0, context_frame_pad, 0))

            context_sample_pad = context_frame_pad * samples_per_frame
            source_signal_slice = nn.functional.pad(source_signal_slice, (context_sample_pad, 0))

        # SOURCE (Wavenet):
        # To make this more tractable, we first apply a µ-law companding transformation
        # (ITU-T, 1988) to the data, and then quantize it to 256 possible values.
        source_signal_slice = mu_law_decode(mu_law_encode(source_signal_slice))
        source_signal_slice = mu_law(source_signal_slice)

        return {
            self.log_mel_spectrogram_prefix: log_mel_spectrogram,  # [num_frames, channels]
            self.signal_prefix: signal,  # [signal_length]
            'source_signal_slice': source_signal_slice,  # [slice_size + receptive_field_size]
            'target_signal_slice': target_signal_slice,  # [slice_size]
            # [(slice_size + receptive_field_size) / samples_per_frame]
            'frames_slice': frames_slice,
        }

    def __getitem__(self, index):
        # log_mel_spectrogram [num_frames, channels]
        log_mel_spectrogram = torch.from_numpy(np.load(
            self.rows[index]['log_mel_spectrogram'])).contiguous()
        # signal [signal_length]
        signal = torch.from_numpy(np.load(self.rows[index]['signal'])).contiguous()
        return self._preprocess(log_mel_spectrogram, signal)

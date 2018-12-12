import logging

from third_party.data_loader import DataLoader as DataBatchLoader
from torch.multiprocessing import cpu_count
from torch.utils import data
from torchnlp.utils import pad_batch

import numpy as np
import random
import torch

from src.hparams import configurable
from src.utils import combine_signal
from src.utils import split_signal
from src.utils import RandomSampler

logger = logging.getLogger(__name__)


class DataLoader(data.Dataset):
    """ DataLoader loading spectrogram slices and audio slices in order to train WaveRNN.

    Args:
        data (iteratable of dict)
        audio_path_key (str, optional): Given a ``dict`` example, this is the key for the
            audio path.
        spectrogram_path_key (str, optional)
        slice_size (int, optional): In spectrogram frames, size of slice.
        slice_pad (int, optional): Pad the spectrogram slice with ``frame_pad`` frames on each side.
        random (random.Random, optional): Random number generator to sample data.

    References:
        * Parallel WaveNet https://arxiv.org/pdf/1711.10433.pdf
          "each containing 7,680 timesteps (roughly 320ms)."
    """

    @configurable
    def __init__(self,
                 data,
                 audio_path_key='aligned_audio_path',
                 spectrogram_path_key='predicted_spectrogram_path',
                 slice_size=3,
                 slice_pad=0,
                 random=random):
        self.data = data
        self.audio_path_key = audio_path_key
        self.spectrogram_path_key = spectrogram_path_key
        self.slice_size = slice_size
        self.slice_pad = slice_pad
        self.random = random

    def __len__(self):
        return len(self.data)

    def _get_slice(self, spectrogram, signal):
        """ Slice the data into bite sized chunks that fit onto GPU memory for training.

        Notes:
            * Frames batch needs to line up with the target signal. Each frame, is used to predict
              the target. The source signal is inputted to predict the target signal; therefore,
              the source signal is one timestep behind.

        Args:
            spectrogram (torch.Tensor [num_frames, channels])
            signal (torch.Tensor [signal_length])

        Returns:
            input_signal (torch.Tensor [signal_length, 2])
            spectrogram_slice (torch.Tensor [num_frames, channels])
            target_signal_coarse (torch.Tensor [signal_length])
            target_signal_fine (torch.Tensor [signal_length])
        """
        samples, num_frames = signal.shape[0], spectrogram.shape[0]
        samples_per_frame = int(samples / num_frames)

        # Signal model requires that there is a scaling factor between the signal and frames
        assert samples % num_frames == 0

        # Get a spectrogram slice
        # ``-slice_size + 1, num_frames - 1`` to ensure there is an equal chance to that a
        # sample will be included inside the slice.
        # For example, with signal ``[1, 2, 3]`` and a ``slice_samples`` of 2 you'd get slices of:
        # (1), (1, 2), (2, 3), (3).
        # With each number represented twice.
        start_frame = self.random.randint(-self.slice_size + 1, num_frames - 1)
        end_frame = min(start_frame + self.slice_size, num_frames)
        start_frame = max(start_frame, 0)

        # Apply padding to slice
        padded_start_frame = max(start_frame - self.slice_pad, 0)
        padded_end_frame = min(end_frame + self.slice_pad, num_frames)
        left_zero_pad = max(-1 * (start_frame - self.slice_pad), 0)
        right_zero_pad = max(end_frame + self.slice_pad - num_frames, 0)

        if self.slice_pad == 0:
            assert left_zero_pad == 0 and right_zero_pad == 0

        spectrogram_slice = spectrogram[padded_start_frame:padded_end_frame]
        spectrogram_slice = torch.nn.functional.pad(spectrogram_slice,
                                                    (0, 0, left_zero_pad, right_zero_pad))

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
            'spectrogram': spectrogram_slice,
            'target_signal_coarse': target_signal_coarse,
            'target_signal_fine': target_signal_fine
        }

    def __getitem__(self, index):
        # signal [signal_length]
        signal = np.load(str(self.data[index][self.audio_path_key]))
        signal = torch.from_numpy(signal).contiguous()
        signal = combine_signal(*split_signal(signal))  # NOTE: Introduce quantization noise

        # log_mel_spectrogram [num_frames, channels]
        spectrogram = np.load(str(self.data[index][self.spectrogram_path_key]))
        spectrogram = torch.from_numpy(spectrogram).contiguous()

        slice_ = self._get_slice(spectrogram, signal)

        return {'slice': slice_, 'spectrogram': spectrogram, 'signal': signal}


class DataBatchIterator(object):
    """ Get a batch iterator over the ``data``.

    Args:
        data (iterable): Data to iterate over.
        batch_size (int): Iteration batch size.
        device (torch.device, optional): Device onto which to load data.
        trial_run (bool or int): If ``True``, iterates over one batch.
        num_workers (int, optional): Number of workers for data loading.
        random (random.Random, optional): Random number generator to sample data.
    """

    def __init__(self,
                 data,
                 batch_size,
                 device,
                 trial_run=False,
                 num_workers=cpu_count(),
                 random=random):
        data = DataLoader(data, random=random)
        # ``drop_last`` to ensure full utilization of mutliple GPUs
        self.iterator = DataBatchLoader(
            data,
            batch_size=batch_size,
            sampler=RandomSampler(data, random=random),
            collate_fn=self._collate_fn,
            pin_memory=True,
            num_workers=num_workers,
            drop_last=True)
        self.device = device
        self.trial_run = trial_run

    def _maybe_cuda(self, tensor, **kwargs):
        return tensor.cuda(device=self.device, **kwargs) if self.device.type == 'cuda' else tensor

    def _collate_fn(self, batch):
        """ Collage function to turn a list of tensors into one batch tensor.

        Returns: (dict) with:
            {
              slice: {
                  input_signal (torch.FloatTensor [batch_size, signal_length])
                  target_signal_coarse  (torch.LongTensor [batch_size, signal_length])
                  target_signal_fine  (torch.LongTensor [batch_size, signal_length])
                  spectrogram (torch.FloatTensor [batch_size, num_frames, frame_channels])
                  spectrogram_lengths (list of int)
                  signal_mask (torch.FloatTensor [batch_size, signal_length])
                  signal_lengths (list of int)
              },
              spectrogram (list of torch.FloatTensor [num_frames, frame_channels])
              signal (list of torch.FloatTensor [signal_length])
            }
        """
        input_signal, signal_lengths = pad_batch([r['slice']['input_signal'] for r in batch])
        target_signal_coarse, _ = pad_batch([r['slice']['target_signal_coarse'] for r in batch])
        target_signal_fine, _ = pad_batch([r['slice']['target_signal_fine'] for r in batch])
        spectrogram, spectrogram_lengths = pad_batch([r['slice']['spectrogram'] for r in batch])

        signal_mask = [torch.full((length,), 1) for length in signal_lengths]
        signal_mask, _ = pad_batch(signal_mask, padding_index=0)  # [batch_size, signal_length]

        return {
            'slice': {
                'input_signal': input_signal,
                'target_signal_coarse': target_signal_coarse,
                'target_signal_fine': target_signal_fine,
                'spectrogram': spectrogram,
                'spectrogram_lengths': spectrogram_lengths,
                'signal_mask': signal_mask,
                'signal_lengths': signal_lengths,
            },
            'spectrogram': [r['spectrogram'] for r in batch],
            'signal': [r['signal'] for r in batch]
        }

    def __len__(self):
        return 1 if self.trial_run else len(self.iterator)

    def __iter__(self):
        for batch in self.iterator:
            maybe_cuda_keys = [
                'spectrogram', 'target_signal_coarse', 'target_signal_fine', 'input_signal',
                'signal_mask'
            ]
            for key in maybe_cuda_keys:
                batch['slice'][key] = self._maybe_cuda(batch['slice'][key], non_blocking=True)

            yield batch

            if self.trial_run:
                break

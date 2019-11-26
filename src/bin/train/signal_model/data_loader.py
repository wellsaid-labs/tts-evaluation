from collections import namedtuple
from functools import partial

import hashlib
import json
import logging
import random

from hparams import configurable
from torch.multiprocessing import cpu_count
from torch.utils.data.sampler import BatchSampler
from torchnlp.samplers import BalancedSampler
from torchnlp.samplers import DeterministicSampler
from torchnlp.samplers import DistributedBatchSampler
from torchnlp.utils import collate_tensors
from torchnlp.utils import tensors_to

import torch

from src.audio import combine_signal
from src.audio import split_signal
from src.environment import IS_TESTING_ENVIRONMENT
from src.utils import maybe_load_tensor

import src
import src.distributed

logger = logging.getLogger(__name__)

SignalModelTrainingRow = namedtuple('SignalModelTrainingRow', [
    'input_signal', 'input_spectrogram', 'target_signal_coarse', 'target_signal_fine', 'signal_mask'
])


def _get_slice(spectrogram, signal, split_signal_partial, spectrogram_slice_size,
               spectrogram_slice_pad):
    """ Slice the data into bite sized chunks that fit onto GPU memory for training.

    Notes:
        * Frames batch needs to line up with the target signal. Each frame, is used to predict
          the target. The source signal is inputted to predict the target signal; therefore,
          the source signal is one timestep behind.

    Args:
        spectrogram (torch.Tensor [num_frames, channels])
        signal (torch.Tensor [signal_length])
        split_signal_partial (callable)
        spectrogram_slice_size (int): In spectrogram frames, size of slice.
        spectrogram_slice_pad (int): Pad the spectrogram slice with ``frame_pad`` frames on each
            side.

    Returns: (SignalModelTrainingRow) (
        input_signal (torch.Tensor [signal_length, 2])
        input_spectrogram (torch.Tensor [num_frames, channels])
        target_signal_coarse (torch.Tensor [signal_length])
        target_signal_fine (torch.Tensor [signal_length])
        signal_mask (torch.BoolTensor [signal_length])
    )
    """
    samples, num_frames = signal.shape[0], spectrogram.shape[0]
    samples_per_frame = int(samples / num_frames)

    # Signal model requires that there is a scaling factor between the signal and frames
    assert samples % num_frames == 0

    # Get a source sample slice shifted back one and target signal
    go_sample = signal.new_zeros(1)  # First sample passed in to start RNN
    source_signal = torch.cat((go_sample, signal), dim=0)
    target_signal = signal
    signal_mask = torch.ones(signal.shape[0], dtype=torch.bool, device=signal.device)

    # Pad spectrogram and signal
    spectrogram_zeros = spectrogram_slice_size - 1 + spectrogram_slice_pad
    spectrogram = torch.nn.functional.pad(spectrogram, (0, 0, spectrogram_zeros, spectrogram_zeros))

    signal_zeros = (spectrogram_slice_size - 1) * samples_per_frame
    target_signal = torch.nn.functional.pad(target_signal, (signal_zeros, signal_zeros))
    source_signal = torch.nn.functional.pad(source_signal, (signal_zeros, signal_zeros - 1))
    signal_mask = torch.nn.functional.pad(signal_mask, (signal_zeros, signal_zeros))

    # Get a spectrogram slice
    # ``-spectrogram_slice_size + 1, num_frames - 1`` to ensure there is an equal chance to that a
    # sample will be included inside the slice.
    # For example, with signal ``[1, 2, 3]`` and a ``slice_samples`` of 2 you'd get slices of:
    # (1), (1, 2), (2, 3), (3).
    # With each number represented twice.
    start_frame = random.randint(-spectrogram_slice_size + 1, num_frames - 1)
    end_frame = (start_frame + spectrogram_slice_size)

    # Get slices from the padded tensors
    # Offset start frame and end frame with `spectrogram_zeros` padding. Increase the range
    # size with `spectrogram_slice_pad` on both ends
    spectrogram_slice = slice(start_frame + spectrogram_zeros - spectrogram_slice_pad,
                              end_frame + spectrogram_zeros + spectrogram_slice_pad)
    spectrogram_slice = spectrogram[spectrogram_slice]

    # Change units from frames to signals and offset with `signal_zeros`
    signal_slice = slice(start_frame * samples_per_frame + signal_zeros,
                         end_frame * samples_per_frame + signal_zeros)
    source_signal_slice = source_signal[signal_slice]
    target_signal_slice = target_signal[signal_slice]
    signal_mask_slice = signal_mask[signal_slice]

    assert source_signal_slice.shape[0] / samples_per_frame == spectrogram_slice_size
    assert target_signal_slice.shape[0] / samples_per_frame == spectrogram_slice_size
    assert spectrogram_slice.shape[0] == spectrogram_slice_size + spectrogram_slice_pad * 2
    # Source is shifted one back from target
    assert torch.equal(source_signal[1:], target_signal[:-1])

    source_signal_coarse_slice, source_signal_fine_slice = split_signal_partial(source_signal_slice)
    target_signal_coarse_slice, target_signal_fine_slice = split_signal_partial(target_signal_slice)

    input_signal_slice = torch.stack((source_signal_coarse_slice, source_signal_fine_slice), dim=1)

    return SignalModelTrainingRow(
        input_signal=input_signal_slice,
        input_spectrogram=spectrogram_slice,
        target_signal_coarse=target_signal_coarse_slice,
        target_signal_fine=target_signal_fine_slice,
        signal_mask=signal_mask_slice)


def _load_fn(row, use_predicted, split_signal_partial, combine_signal_partial, **kwargs):
    """ Load function for loading a single row.

    Args:
        row (TextSpeechRow)
        use_predicted (bool): If ``True`` use predicted spectrogram as opposed to the real one.
        split_signal_partial (callable): `src.audio.split_signal` configured partial.
        combine_signal_partial (callable): `src.audio.combine_signal` configured partial.

    Returns:
        (SignalModelTrainingRow)
    """
    spectrogram = maybe_load_tensor(row.predicted_spectrogram if use_predicted else row.spectrogram)
    spectrogram_audio = maybe_load_tensor(row.spectrogram_audio).float()
    spectrogram_audio = combine_signal_partial(*split_signal_partial(spectrogram_audio))

    # Check invariants
    assert spectrogram.shape[0] > 0
    assert spectrogram_audio.shape[0] > 0

    return _get_slice(spectrogram, spectrogram_audio, split_signal_partial, **kwargs)


class DataLoader(src.utils.DataLoader):
    """ Get a batch iterator over the ``data``.

    Args:
        data (iterable of TextSpeechRow): Data to iterate over.
        batch_size (int): Iteration batch size.
        device (torch.device): Device onto which to load data.
        use_predicted (bool): If ``True`` use predicted spectrogram as opposed to the real one.
        num_workers (int): Number of workers used to load data.
        **kwargs (any): Other arguments to the data loader ``_load_fn``

    Returns:
        Single-process or multi-process iterators over the dataset. Per iteration the batch returned
        includes: SpectrogramModelTrainingRow (
            input_signal (torch.FloatTensor
                [batch_size, spectrogram_slice_size * samples_per_frame])
            input_spectrogram (torch.FloatTensor
                [batch_size, spectrogram_slice_size + spectrogram_slice_pad, frame_channels])
            target_signal_coarse (torch.LongTensor
                [batch_size, spectrogram_slice_size * samples_per_frame])
            target_signal_fine (torch.LongTensor
                [batch_size, spectrogram_slice_size * samples_per_frame])
            signal_mask (torch.BoolTensor
                [batch_size, spectrogram_slice_size * samples_per_frame])
        )
    """

    @configurable
    def __init__(self,
                 data,
                 batch_size,
                 device,
                 use_predicted,
                 num_workers=0 if IS_TESTING_ENVIRONMENT else cpu_count(),
                 **kwargs):
        if src.distributed.is_initialized():
            # NOTE: `DistributedBatchSampler` assumes that the workers and master have the same
            # sampling; therefore, the same data.
            # NOTE: Learn more about `hashlib` and `json` here:
            # https://stackoverflow.com/questions/5417949/computing-an-md5-hash-of-a-data-structure
            hash_ = hashlib.md5(json.dumps([e.text for e in data]).encode('utf-8')).hexdigest()
            src.distributed.assert_synced(
                int(hash_, 16), 'This dataset does not match the master dataset.')

        sampler = BalancedSampler(
            data,
            get_class=lambda e: e.speaker,
            get_weight=lambda e: (e.predicted_spectrogram
                                  if use_predicted else e.spectrogram).shape[0])
        # NOTE: `drop_last` to ensure full utilization of mutliple GPUs.
        batch_sampler = BatchSampler(sampler, batch_size, drop_last=True)
        batch_sampler = DeterministicSampler(batch_sampler)

        if src.distributed.is_initialized():
            num_workers = int(num_workers / torch.distributed.get_world_size())
            batch_sampler = DistributedBatchSampler(batch_sampler)

        super().__init__(
            data,
            collate_fn=collate_tensors,
            load_fn=partial(
                _load_fn,
                use_predicted=use_predicted,
                split_signal_partial=split_signal.get_configured_partial(),
                combine_signal_partial=combine_signal.get_configured_partial(),
                **kwargs),
            pin_memory=True,
            post_processing_fn=partial(tensors_to, device=device, non_blocking=True),
            batch_sampler=batch_sampler,
            num_workers=num_workers)

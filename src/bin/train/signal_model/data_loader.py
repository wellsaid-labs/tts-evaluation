from collections import defaultdict
from collections import namedtuple
from functools import partial

import hashlib
import json
import logging
import random

from hparams import configurable
from torch.multiprocessing import cpu_count
from torch.utils.data.sampler import BatchSampler
from torchnlp._third_party.weighted_random_sampler import WeightedRandomSampler
from torchnlp.samplers import DeterministicSampler
from torchnlp.samplers import DistributedBatchSampler
from torchnlp.utils import collate_tensors
from torchnlp.utils import tensors_to

import torch

from src.audio import to_floating_point_pcm
from src.environment import IS_TESTING_ENVIRONMENT
from src.utils import maybe_load_tensor

import src
import src.distributed

logger = logging.getLogger(__name__)

SignalModelTrainingRow = namedtuple(
    'SignalModelTrainingRow', ['spectrogram', 'spectrogram_mask', 'target_signal', 'signal_mask'])


class _BalancedSampler(WeightedRandomSampler):
    """ Weighted sampler with respect for an element's class.

    Args:
        data (iterable)
        get_class (callable, optional): Get the class of an item relative to the entire dataset.
        get_weight (callable, optional): Define a weight for each item other than one.
        kwargs: Additional key word arguments passed onto `WeightedRandomSampler`.
    """

    def __init__(self, data_source, get_class, get_weight, **kwargs):
        classified = [get_class(item) for item in data_source]
        weighted = [float(get_weight(item)) for item in data_source]
        totals = defaultdict(float)
        for class_, weight in zip(classified, weighted):
            totals[class_] += weight**2
        weights = [(w / totals[c]) if w > 0 else 0.0 for c, w in zip(classified, weighted)]
        super().__init__(weights=weights, **kwargs)


def _get_slice(spectrogram, signal, spectrogram_slice_size, spectrogram_slice_pad):
    """ Slice the data into bite sized chunks that fit onto GPU memory for training.

    Notes:
        * Frames batch needs to line up with the target signal. Each frame, is used to predict
          the target.

    Args:
        spectrogram (torch.FloatTensor [num_frames, channels])
        signal (torch.FloatTensor [signal_length])
        spectrogram_slice_size (int): In spectrogram frames, size of slice.
        spectrogram_slice_pad (int): Pad the spectrogram slice with ``frame_pad`` frames on each
            side.

    Returns: (SignalModelTrainingRow) (
        spectrogram (torch.FloatTensor [num_frames, channels])
        spectrogram_mask (torch.BoolTensor [num_frames])
        target_signal (torch.FloatTensor [signal_length])
        signal_mask (torch.BoolTensor [signal_length])
    )
    """
    samples, num_frames = signal.shape[0], spectrogram.shape[0]
    samples_per_frame = int(samples / num_frames)

    # Signal model requires that there is a scaling factor between the signal and frames
    assert samples % num_frames == 0

    signal_mask = torch.ones(signal.shape[0], dtype=torch.bool, device=signal.device)
    spectrogram_mask = torch.ones(spectrogram.shape[0], dtype=torch.bool, device=spectrogram.device)

    # Pad spectrogram and signal
    spectrogram_zeros = spectrogram_slice_size - 1 + spectrogram_slice_pad
    spectrogram = torch.nn.functional.pad(spectrogram, (0, 0, spectrogram_zeros, spectrogram_zeros))
    spectrogram_mask = torch.nn.functional.pad(spectrogram_mask,
                                               (spectrogram_zeros, spectrogram_zeros))

    assert spectrogram_mask.shape[0] == spectrogram.shape[0]

    signal_zeros = (spectrogram_slice_size - 1) * samples_per_frame
    target_signal = torch.nn.functional.pad(signal, (signal_zeros, signal_zeros))
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
    spectrogram_mask_slice = spectrogram_mask[spectrogram_slice]
    spectrogram_slice = spectrogram[spectrogram_slice]

    # Change units from frames to signals and offset with `signal_zeros`
    signal_slice = slice(start_frame * samples_per_frame + signal_zeros,
                         end_frame * samples_per_frame + signal_zeros)
    target_signal_slice = target_signal[signal_slice]
    signal_mask_slice = signal_mask[signal_slice]

    assert target_signal_slice.shape[0] / samples_per_frame == spectrogram_slice_size
    assert spectrogram_slice.shape[0] == spectrogram_slice_size + spectrogram_slice_pad * 2
    assert spectrogram_mask_slice.shape[0] == spectrogram_slice.shape[0]
    assert target_signal_slice.shape == signal_mask_slice.shape

    return SignalModelTrainingRow(
        spectrogram=spectrogram_slice,
        spectrogram_mask=spectrogram_mask_slice,
        target_signal=target_signal_slice,
        signal_mask=signal_mask_slice)


def _load_fn(row, use_predicted, **kwargs):
    """ Load function for loading a single `SignalModelTrainingRow` row from `TextSpeechRow`.

    Args:
        row (TextSpeechRow)
        use_predicted (bool): If ``True`` use predicted spectrogram as opposed to the real one.
        **kwargs: Key word arguments passed to `_get_slice`.

    Returns:
        (SignalModelTrainingRow)
    """
    spectrogram = maybe_load_tensor(row.predicted_spectrogram if use_predicted else row.spectrogram)
    # NOTE: `row.spectrogram_audio` is a `torch.ShortTensor` (16-bit integer) while our model
    # requires a `torch.FloatTensor` (32-bit floating point)
    spectrogram_audio = to_floating_point_pcm(maybe_load_tensor(row.spectrogram_audio))

    # Check invariants
    assert spectrogram.shape[0] > 0
    assert spectrogram_audio.shape[0] > 0

    return _get_slice(spectrogram, spectrogram_audio, **kwargs)


class DataLoader(src.utils.DataLoader):
    """ Get a batch iterator over the ``data``.

    Args:
        data (iterable of TextSpeechRow): Data to iterate over.
        batch_size (int): Iteration batch size.
        device (torch.device): Device onto which to load data.
        use_predicted (bool): If ``True`` use predicted spectrogram as opposed to the real one.
        num_workers (int): Number of workers used to load data.
        max_workers_per_process (int, optional): The maximum workers per process used for data
            loading.
        balance_speaker (bool): If `True` this equalizes the audio data sampled for each speaker.
        **kwargs (any): Other arguments to the data loader ``_load_fn``

    Returns:
        Single-process or multi-process iterators over the dataset. Per iteration the batch returned
        includes: SpectrogramModelTrainingRow (
            spectrogram (torch.FloatTensor
                [batch_size, spectrogram_slice_size + spectrogram_slice_pad, frame_channels])
            spectrogram_mask (torch.BoolTensor
                [batch_size, spectrogram_slice_size + spectrogram_slice_pad])
            target_signal (torch.FloatTensor
                [batch_size, spectrogram_slice_size * samples_per_frame])
            signal_mask (torch.BoolTensor [batch_size, spectrogram_slice_size * samples_per_frame])
        )
    """

    @configurable
    def __init__(self,
                 data,
                 batch_size,
                 device,
                 use_predicted,
                 num_workers=0 if IS_TESTING_ENVIRONMENT else cpu_count(),
                 max_workers_per_process=6,
                 **kwargs):
        world_size = torch.distributed.get_world_size() if src.distributed.is_initialized() else 1
        num_workers = min(num_workers, max_workers_per_process * world_size)

        if src.distributed.is_initialized():
            # NOTE: `DistributedBatchSampler` assumes that the workers and master have the same
            # sampling; therefore, the same data.
            # NOTE: Learn more about `hashlib` and `json` here:
            # https://stackoverflow.com/questions/5417949/computing-an-md5-hash-of-a-data-structure
            hash_ = hashlib.md5(json.dumps([e.text for e in data]).encode('utf-8')).hexdigest()
            src.distributed.assert_synced(
                int(hash_, 16), 'This dataset does not match the master dataset.')

        sampler = _BalancedSampler(
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
            load_fn=partial(_load_fn, use_predicted=use_predicted, **kwargs),
            pin_memory=True,
            post_processing_fn=partial(tensors_to, device=device, non_blocking=True),
            batch_sampler=batch_sampler,
            num_workers=num_workers)

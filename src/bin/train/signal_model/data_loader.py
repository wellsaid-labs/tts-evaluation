from collections import namedtuple
from functools import partial

import logging

import random
import torch

from src.audio import combine_signal
from src.audio import split_signal
from src.hparams import configurable
from src.utils import collate_sequences
from src.utils import DataLoader
from src.utils import RandomSampler
from src.utils import tensors_to
from src.utils import OnDiskTensor

logger = logging.getLogger(__name__)

SignalModelTrainingRow = namedtuple('SignalModelTrainingRow', [
    'input_signal', 'input_spectrogram', 'target_signal_coarse', 'target_signal_fine', 'signal_mask'
])


def _get_slice(spectrogram, signal, slice_size, slice_pad, random=random):
    """ Slice the data into bite sized chunks that fit onto GPU memory for training.

    Notes:
        * Frames batch needs to line up with the target signal. Each frame, is used to predict
          the target. The source signal is inputted to predict the target signal; therefore,
          the source signal is one timestep behind.

    Args:
        spectrogram (torch.Tensor [num_frames, channels])
        signal (torch.Tensor [signal_length])
        slice_size (int): In spectrogram frames, size of slice.
        slice_pad (int): Pad the spectrogram slice with ``frame_pad`` frames on each side.
        random (random.Random, optional): Random number generator to sample data.

    Returns: (SignalModelTrainingRow) (
        input_signal (torch.Tensor [signal_length, 2])
        input_spectrogram (torch.Tensor [num_frames, channels])
        target_signal_coarse (torch.Tensor [signal_length])
        target_signal_fine (torch.Tensor [signal_length])
        signal_mask (torch.Tensor [signal_length])
    )
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
    start_frame = random.randint(-slice_size + 1, num_frames - 1)
    end_frame = min(start_frame + slice_size, num_frames)
    start_frame = max(start_frame, 0)

    # Apply padding to slice
    padded_start_frame = max(start_frame - slice_pad, 0)
    padded_end_frame = min(end_frame + slice_pad, num_frames)
    left_zero_pad = max(-1 * (start_frame - slice_pad), 0)
    right_zero_pad = max(end_frame + slice_pad - num_frames, 0)

    if slice_pad == 0:
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
    signal_length = target_signal_coarse.shape[0]
    signal_mask = torch.full((signal_length,), 1).byte()

    return SignalModelTrainingRow(
        input_signal=input_signal,
        input_spectrogram=spectrogram_slice,
        target_signal_coarse=target_signal_coarse,
        target_signal_fine=target_signal_fine,
        signal_mask=signal_mask)


def _load_fn(row, use_predicted, **kwargs):
    """ Load function for loading a single row.

    Args:
        row (SpectrogramTextSpeechRow)
        use_predicted (bool): If ``True`` use predicted spectrogram as opposed to the real one.

    Returns:
        (SignalModelTrainingRow)
    """
    spectrogram = row.predicted_spectrogram if use_predicted else row.spectrogram
    spectrogram = spectrogram.to_tensor() if isinstance(spectrogram, OnDiskTensor) else spectrogram
    spectrogram_audio = row.spectrogram_audio.to_tensor() if isinstance(
        row.spectrogram_audio, OnDiskTensor) else row.spectrogram_audio
    spectrogram_audio = combine_signal(*split_signal(spectrogram_audio))

    # Check invariants
    assert spectrogram.shape[0] > 0
    assert spectrogram_audio.shape[0] > 0

    return _get_slice(spectrogram, spectrogram_audio, **kwargs)


class DataLoader(DataLoader):
    """ Get a batch iterator over the ``data``.

    Args:
        data (iterable of SpectrogramTextSpeechRow): Data to iterate over.
        batch_size (int): Iteration batch size.
        device (torch.device): Device onto which to load data.
        use_tqdm (bool): If ``True`` display progress via TQDM.
        trial_run (bool or int): If ``True``, iterates over one batch.
        random (random.Random, optional): Random number generator to sample data.
        **kwargs (any): Other arguments to the data loader ``_load_fn``

    Returns:
        Single-process or multi-process iterators over the dataset. Per iteration the batch returned
        includes: SpectrogramModelTrainingRow (
            input_signal (tuple(torch.FloatTensor [batch_size, signal_length],
                                torch.LongTensor [batch_size]))
            input_spectrogram (tuple(torch.FloatTensor [batch_size, num_frames, frame_channels]),
                                     torch.LongTensor [batch_size]))
            target_signal_coarse (tuple(torch.LongTensor [batch_size, signal_length],
                                        torch.LongTensor [batch_size]))
            target_signal_fine (tuple(torch.LongTensor [batch_size, signal_length]),
                                      torch.LongTensor [batch_size]))
            signal_mask (tuple(torch.FloatTensor [batch_size, signal_length],
                               torch.LongTensor [batch_size]))
        )
    """

    @configurable
    def __init__(self, data, batch_size, device, use_tqdm, trial_run, random=random, **kwargs):

        # ``drop_last`` to ensure full utilization of mutliple GPUs
        super().__init__(
            data,
            batch_size=batch_size,
            collate_fn=partial(collate_sequences),
            drop_last=True,
            load_fn=partial(_load_fn, random=random, **kwargs),
            pin_memory=True,
            post_processing_fn=partial(tensors_to, device=device, non_blocking=True),
            sampler=RandomSampler(data, random=random),
            trial_run=trial_run,
            use_tqdm=use_tqdm)

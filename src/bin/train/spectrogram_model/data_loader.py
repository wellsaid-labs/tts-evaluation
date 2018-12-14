from collections import namedtuple
from functools import partial

import logging

from torch.multiprocessing import cpu_count
from torchnlp.samplers import BucketBatchSampler

import torch

from src.utils import collate_sequences
from src.utils import DataLoader
from src.utils import get_spectrogram_lengths
from src.utils import identity
from src.utils import OnDiskTensor
from src.utils import tensors_to

import src.distributed

logger = logging.getLogger(__name__)

SpectrogramModelTrainingRow = namedtuple('SpectrogramModelTrainingRow', [
    'text', 'speaker', 'spectrogram', 'stop_token', 'spectrogram_mask', 'spectrogram_expanded_mask'
])


def _load_fn(row, text_encoder, speaker_encoder):
    """ Load function for loading a single row.

    Args:
        row (SpectrogramTextSpeechRow)
        text_encoder (torchnlp.TextEncoder): Text encoder used to encode and decode the text.
        speaker_encoder (torchnlp.TextEncoder): Text encoder used to encode and decode the speaker.

    Returns:
        (SpectrogramModelTrainingRow)
    """
    spectrogram = row.spectrogram.to_tensor() if isinstance(row.spectrogram,
                                                            OnDiskTensor) else row.spectrogram
    stop_token = spectrogram.new_zeros((spectrogram.shape[0],))
    stop_token[-1] = 1

    # Check invariants
    assert len(row.text) > 0
    assert spectrogram.shape[0] > 0

    return SpectrogramModelTrainingRow(
        text=text_encoder.encode(row.text),
        speaker=speaker_encoder.encode(row.speaker),
        stop_token=stop_token,
        spectrogram=spectrogram,
        spectrogram_mask=torch.FloatTensor(spectrogram.shape[0]).fill_(1),
        spectrogram_expanded_mask=torch.FloatTensor(*spectrogram.shape).fill_(1))


class DataLoader(DataLoader):
    """ Get a batch iterator over the ``data``.

    Args:
        data (iterable): Data to iterate over.
        batch_size (int): Iteration batch size.
        device (torch.device): Device onto which to load data.
        use_tqdm (bool): If ``True`` display progress via TQDM.
        trial_run (bool or int): If ``True``, iterates over one batch.
        num_workers (int, optional): Number of workers for data loading.
        **kwargs (any): Other arguments to the data loader ``_load_fn``

    Returns:
        Single-process or multi-process iterators over the dataset. Per iteration the batch returned
        includes: SpectrogramModelTrainingRow (
            text (tuple(torch.LongTensor [num_tokens, batch_size], list of int))
            spectrogram (tuple(torch.FloatTensor [num_frames, batch_size, frame_channels],
                              list of int))
            stop_token (tuple(torch.FloatTensor [num_frames, batch_size], list of int))
            speaker (tuple(torch.LongTensor [1, batch_size], list of int))
            spectrogram_expanded_mask (tuple(torch.FloatTensor [num_frames, batch_size,
                                                                frame_channels],
                                             list of int))
            spectrogram_mask (tuple(torch.FloatTensor [num_frames, batch_size], list of int))
        )
    """

    def __init__(self,
                 data,
                 batch_size,
                 device,
                 use_tqdm,
                 trial_run=False,
                 num_workers=cpu_count(),
                 **kwargs):
        batch_sampler = None
        if not src.distributed.in_use() or src.distributed.is_master():
            batch_sampler = BucketBatchSampler(
                get_spectrogram_lengths(data),
                batch_size,
                drop_last=True,
                sort_key=identity,
                biggest_batches_first=identity)

        if src.distributed.in_use():
            # Given there are multiple processes, we change the data loaded per process.
            num_workers = int(num_workers / torch.distributed.get_world_size())
            batch_sampler = src.distributed.distribute_batch_sampler(batch_sampler, batch_size,
                                                                     device)

        super().__init__(
            data,
            load_fn=partial(_load_fn, **kwargs),
            post_processing_fn=partial(tensors_to, device=device, non_blocking=True),
            batch_sampler=batch_sampler,
            collate_fn=partial(collate_sequences, dim=1),
            pin_memory=True,
            num_workers=num_workers,
            trial_run=trial_run)

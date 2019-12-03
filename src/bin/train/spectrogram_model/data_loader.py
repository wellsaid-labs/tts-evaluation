from collections import defaultdict
from collections import namedtuple
from functools import partial

import hashlib
import json
import logging

from torch.multiprocessing import cpu_count
from torchnlp._third_party.weighted_random_sampler import WeightedRandomSampler
from torchnlp.encoders.text import stack_and_pad_tensors
from torchnlp.samplers import BucketBatchSampler
from torchnlp.samplers import DeterministicSampler
from torchnlp.samplers import DistributedBatchSampler
from torchnlp.samplers import get_number_of_elements
from torchnlp.samplers import OomBatchSampler
from torchnlp.utils import collate_tensors
from torchnlp.utils import tensors_to

import torch

from src.environment import IS_TESTING_ENVIRONMENT
from src.utils import DataLoader
from src.utils import maybe_load_tensor

import src.distributed

logger = logging.getLogger(__name__)

SpectrogramModelTrainingRow = namedtuple('SpectrogramModelTrainingRow', [
    'text', 'speaker', 'spectrogram', 'stop_token', 'spectrogram_mask', 'spectrogram_expanded_mask'
])


class _BalancedSampler(WeightedRandomSampler):
    """ Weighted sampler with respect for an element's class.

    Args:
        data (iterable)
        get_class (callable): Get the class of an item relative to the entire dataset.
        get_weight (callable): Define a weight for each item other than one.
        kwargs: Additional key word arguments passed onto `WeightedRandomSampler`.
    """

    def __init__(self, data_source, get_class, get_weight, **kwargs):
        classified = [get_class(item) for item in data_source]
        weighted = [float(get_weight(item)) for item in data_source]
        totals = defaultdict(float)
        for class_, weight in zip(classified, weighted):
            totals[class_] += weight
        weights = [1 / totals[c] if w > 0 else 0.0 for c, w in zip(classified, weighted)]
        super().__init__(weights=weights, **kwargs)


def _load_fn(row, input_encoder):
    """ Load function for loading a single row.

    Args:
        row (TextSpeechRow)
        input_encoder (src.spectrogram_model.InputEncoder)

    Returns:
        (SpectrogramModelTrainingRow)
    """
    spectrogram = maybe_load_tensor(row.spectrogram)
    stop_token = spectrogram.new_zeros((spectrogram.shape[0],))
    stop_token[-1] = 1

    # Check invariants
    assert len(row.text) > 0
    assert spectrogram.shape[0] > 0

    text, speaker = input_encoder.encode((row.text, row.speaker))

    return SpectrogramModelTrainingRow(
        text=text,
        speaker=speaker,
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
        num_workers (int, optional): Number of workers for data loading.
        **kwargs (any): Other arguments to the data loader ``_load_fn``

    Returns:
        Single-process or multi-process iterators over the dataset. Per iteration the batch returned
        includes: SpectrogramModelTrainingRow (
            text (BatchedSequences(torch.LongTensor [num_tokens, batch_size],
                        torch.LongTensor [1, batch_size]))
            spectrogram (BatchedSequences(
                          torch.FloatTensor [num_frames, batch_size, frame_channels],
                          torch.LongTensor [1, batch_size]))
            stop_token (BatchedSequences(torch.FloatTensor [num_frames, batch_size],
                              torch.LongTensor [1, batch_size]))
            speaker (BatchedSequences(torch.LongTensor [1, batch_size],
                           torch.LongTensor [1, batch_size]))
            spectrogram_expanded_mask (BatchedSequences(torch.FloatTensor [num_frames, batch_size,
                                                                frame_channels],
                                             torch.LongTensor [1, batch_size]))
            spectrogram_mask (BatchedSequences(torch.FloatTensor [num_frames, batch_size],
                                    torch.LongTensor [1, batch_size]))
        )
    """

    def __init__(self,
                 data,
                 batch_size,
                 device,
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

        # NOTE: The training and development dataset distribution of speakers is arbitrary (i.e.
        # some audio books have more data and some have less). In order to ensure that no speaker
        # is prioritized over another, we balance the number of spectrogram frames.
        sampler = _BalancedSampler(
            data, get_class=lambda e: e.speaker, get_weight=lambda e: e.spectrogram.shape[0])
        batch_sampler = BucketBatchSampler(
            sampler, batch_size, drop_last=True, sort_key=lambda i: data[i].spectrogram.shape[0])
        batch_sampler = OomBatchSampler(
            batch_sampler, get_item_size=lambda i: get_number_of_elements(data[i]))
        batch_sampler = DeterministicSampler(batch_sampler)

        if src.distributed.is_initialized():
            num_workers = int(num_workers / torch.distributed.get_world_size())
            batch_sampler = DistributedBatchSampler(batch_sampler)

        super().__init__(
            data,
            load_fn=partial(_load_fn, **kwargs),
            post_processing_fn=partial(tensors_to, device=device, non_blocking=True),
            batch_sampler=batch_sampler,
            collate_fn=partial(
                collate_tensors, stack_tensors=partial(stack_and_pad_tensors, dim=1)),
            pin_memory=True,
            num_workers=num_workers)

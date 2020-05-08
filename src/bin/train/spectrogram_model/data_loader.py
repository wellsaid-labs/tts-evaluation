from collections import defaultdict
from collections import namedtuple
from functools import partial

import hashlib
import json
import logging

from hparams import configurable
from hparams import HParam
from scipy import ndimage
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

import numpy as np
import torch

from src.environment import IS_TESTING_ENVIRONMENT
from src.utils import DataLoader
from src.utils import maybe_load_tensor

import src.distributed

logger = logging.getLogger(__name__)

SpectrogramModelTrainingRow = namedtuple('SpectrogramModelTrainingRow', [
    'text', 'speaker', 'spectrogram', 'stop_token', 'spectrogram_mask', 'spectrogram_expanded_mask'
])


def _get_normalized_half_gaussian(length, standard_deviation):
    gaussian_kernel = ndimage.gaussian_filter1d(
        np.float_([0] * (length - 1) + [1]), sigma=standard_deviation)
    gaussian_kernel = gaussian_kernel / gaussian_kernel.max()
    return torch.tensor(gaussian_kernel).float()


@configurable
def get_normalized_half_gaussian(length=HParam(), standard_deviation=HParam()):
    """ Get a normalized half guassian distribution.

    Learn more:
    https://en.wikipedia.org/wiki/Half-normal_distribution
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter1d.html

    Args:
        length (int): The size of the gaussian filter.
        standard_deviation (float): The standard deviation of the guassian.

    Returns:
        (torch.FloatTensor [length,])
    """
    return _get_normalized_half_gaussian(length, standard_deviation)


class _BalancedSampler(WeightedRandomSampler):
    """ Ensure each class is sampled uniformly from.

    For example: If `get_weight` is equal to the audio length and `get_class` is equal to the
    speaker, in a TTS dataset, this ensures that an equal amount of audio is sampled per speaker.

    Args:
        data_source (iterable)
        get_class (callable): Get the class of an item relative to the entire dataset.
        get_weight (callable): Define a weight for each item.
        **kwargs: Additional key word arguments passed onto `WeightedRandomSampler`.
    """

    def __init__(self, data_source, get_class, get_weight, **kwargs):
        classified = [get_class(item) for item in data_source]
        weighted = [float(get_weight(item)) for item in data_source]
        totals = defaultdict(float)
        for class_, weight in zip(classified, weighted):
            totals[class_] += weight
        weights = [1 / totals[c] if w > 0 else 0.0 for c, w in zip(classified, weighted)]

        super().__init__(weights=weights, **kwargs)

        # NOTE: Ensure all weights add up to 1.0
        normalized_weights = self.weights / self.weights.sum()
        # NOTE: The average weight of the loaded data.
        self.expected_weight = sum([w * n for w, n in zip(weighted, normalized_weights)])


def _load_fn(row, input_encoder, get_normalized_half_gaussian_partial):
    """ Load function for loading a single row.

    Args:
        row (TextSpeechRow)
        input_encoder (src.spectrogram_model.InputEncoder)
        get_normalized_half_gaussian_partial (callable)

    Returns:
        (SpectrogramModelTrainingRow)
    """
    spectrogram = maybe_load_tensor(row.spectrogram)
    stop_token = spectrogram.new_zeros((spectrogram.shape[0],))

    # NOTE: The exact stop token distribution is uncertain because there are multiple valid
    # stopping points after someone has finished speaking. For example, the audio can be cutoff
    # 1 second or 2 seconds after someone has finished speaking. In order to address this
    # uncertainty, we naively apply a normal distribution as the stop token ground truth.
    # NOTE: This strategy was found to be effective via Comet in January 2020.
    # TODO: In the future, it'd likely be more accurate to base the probability for stopping
    # based on the loudness of each frame. The maximum loudness is based on a full-scale sine wave
    # and the minimum loudness would be -96 Db or so. The probability for stopping is the loudness
    # relative to the minimum and maximum loudness. This is assuming that at the end of an audio
    # clip it gets progressively quieter.
    gaussian_kernel = get_normalized_half_gaussian_partial()
    max_len = min(len(stop_token), len(gaussian_kernel))
    stop_token[-max_len:] = gaussian_kernel[-max_len:]

    # Check invariants
    assert len(row.text) > 0
    assert spectrogram.shape[0] > 0

    text, speaker = input_encoder.encode((row.text, row.speaker))

    return SpectrogramModelTrainingRow(
        text=text,
        speaker=speaker,
        stop_token=stop_token,
        spectrogram=spectrogram,
        spectrogram_mask=torch.ones(spectrogram.shape[0]),
        spectrogram_expanded_mask=torch.ones(*spectrogram.shape))


class DataLoader(DataLoader):
    """ Get a batch iterator over the ``data``.

    Args:
        data (iterable of TextSpeechRow): Data to iterate over.
        batch_size (int): Iteration batch size.
        device (torch.device): Device onto which to load data.
        num_workers (int, optional): Number of workers for data loading.
        max_workers_per_process (int, optional): The maximum workers per process used for data
            loading. This default was set based on the 7b7af914fde844cab49cd1bbb6702315 experiment.
        **kwargs (any): Other arguments to the data loader ``_load_fn``

    Returns:
        Single-process or multi-process iterators over the dataset. Per iteration the batch returned
        includes: SpectrogramModelTrainingRow (
            text (BatchedSequences(torch.LongTensor [num_tokens, batch_size],
                        torch.LongTensor [1, batch_size]))
            speaker (BatchedSequences(torch.LongTensor [1, batch_size],
                           torch.LongTensor [1, batch_size]))
            stop_token (BatchedSequences(torch.FloatTensor [num_frames, batch_size],
                              torch.LongTensor [1, batch_size]))
            spectrogram (BatchedSequences(
                          torch.FloatTensor [num_frames, batch_size, frame_channels],
                          torch.LongTensor [1, batch_size]))
            spectrogram_mask (BatchedSequences(torch.FloatTensor [num_frames, batch_size],
                                    torch.LongTensor [1, batch_size]))
            spectrogram_expanded_mask (BatchedSequences(torch.FloatTensor [num_frames, batch_size,
                                                                frame_channels],
                                             torch.LongTensor [1, batch_size]))
        )
    """

    def __init__(self,
                 data,
                 batch_size,
                 device,
                 num_workers=0 if IS_TESTING_ENVIRONMENT else cpu_count(),
                 max_workers_per_process=4,
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

        # NOTE: The training and development dataset distribution of speakers is arbitrary (i.e.
        # some audio books have more data and some have less). In order to ensure that no speaker
        # is prioritized over another, we balance the number of spectrogram frames per speaker
        sampler = _BalancedSampler(
            data, get_class=lambda e: e.speaker, get_weight=lambda e: e.spectrogram.shape[0])
        batch_sampler = BucketBatchSampler(
            sampler, batch_size, drop_last=True, sort_key=lambda i: data[i].spectrogram.shape[0])
        # TODO: `get_number_of_elements` is not compatible with `OnDiskTensor`, fix that.
        batch_sampler = OomBatchSampler(
            batch_sampler, get_item_size=lambda i: get_number_of_elements(data[i]))
        batch_sampler = DeterministicSampler(batch_sampler)

        if src.distributed.is_initialized():
            num_workers = int(num_workers / torch.distributed.get_world_size())
            batch_sampler = DistributedBatchSampler(batch_sampler)

        self.expected_average_spectrogram_length = sampler.expected_weight

        logger.info('The expected average spectrogram length is: %f',
                    self.expected_average_spectrogram_length)

        super().__init__(
            data,
            load_fn=partial(
                _load_fn,
                get_normalized_half_gaussian_partial=get_normalized_half_gaussian
                .get_configured_partial(),
                **kwargs),
            post_processing_fn=partial(tensors_to, device=device, non_blocking=True),
            batch_sampler=batch_sampler,
            collate_fn=partial(
                collate_tensors, stack_tensors=partial(stack_and_pad_tensors, dim=1)),
            pin_memory=True,
            num_workers=num_workers)

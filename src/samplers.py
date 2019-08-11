from contextlib import contextmanager

import heapq
import math

from third_party.samplers import WeightedRandomSampler
from torch.utils.data.sampler import BatchSampler
from torch.utils.data.sampler import Sampler
from torch.utils.data.sampler import SubsetRandomSampler
from torchnlp.samplers.sorted_sampler import SortedSampler
from torchnlp.utils import get_tensors

import torch
import torch.distributed

from src.environment import fork_rng
from src.environment import get_initial_seed
from src.environment import set_seed
from src.environment import get_random_generator_state
from src.environment import set_random_generator_state
from src.utils import identity

import src.distributed


class BalancedSampler(WeightedRandomSampler):
    """ Samples uniformly from each class.

    Args:
        data (iterable)
        get_class (callable, optional): Get the class of an item relative to the entire dataset.
        get_weight (callable, optional): Optionally, define a weight for each item other than one.
        kwargs: Additional key word arguments passed onto `WeightedRandomSampler`.
    """

    def __init__(self, data_source, get_class=identity, get_weight=lambda x: 1, **kwargs):
        classified = [get_class(item) for item in data_source]
        weighted = [float(get_weight(item)) for item in data_source]
        class_totals = {
            k: sum([w for c, w in zip(classified, weighted) if k == c]) for k in set(classified)
        }
        weights = [w / class_totals[c] if w > 0 else 0.0 for c, w in zip(classified, weighted)]
        if 'num_samples' not in kwargs:
            kwargs['num_samples'] = len(data_source)
        super().__init__(weights=weights, **kwargs)


class BucketBatchSampler(BatchSampler):
    """ `BucketBatchSampler` toggles between `sampler` batches and sorted batches.

    Typically, the `sampler` will be a `RandomSampler` allowing the user to toggle between
    random batches and sorted batches. A larger `bucket_size_multiplier` is more sorted and vice
    versa.

    Background:
        `BucketBatchSampler` is similar to a `BucketIterator` found in popular libraries like
        `AllenNLP` and `torchtext`. A `BucketIterator` pools together examples with a similar size
        length to reduce the padding required for each batch while maintaining some noise through
        bucketing.

        **AllenNLP Implementation:**
        https://github.com/allenai/allennlp/blob/master/allennlp/data/iterators/bucket_iterator.py

        **torchtext Implementation:**
        https://github.com/pytorch/text/blob/master/torchtext/data/iterator.py#L225

    Args:
        sampler (torch.data.utils.sampler.Sampler):
        batch_size (int): Size of mini-batch.
        drop_last (bool): If `True` the sampler will drop the last batch if its size would be less
            than `batch_size`.
        sort_key (callable, optional): Callable to specify a comparison key for sorting.
        bucket_size_multiplier (int, optional): Buckets are of size
            `batch_size * bucket_size_multiplier`.
    """

    def __init__(self,
                 sampler,
                 batch_size,
                 drop_last,
                 sort_key=identity,
                 bucket_size_multiplier=100):
        super().__init__(sampler, batch_size, drop_last)
        self.sort_key = sort_key
        self.bucket_sampler = BatchSampler(sampler, batch_size * bucket_size_multiplier, False)

    def __iter__(self):
        for bucket in self.bucket_sampler:
            sorted_sampler = SortedSampler(bucket, self.sort_key)
            for batch in SubsetRandomSampler(
                    list(BatchSampler(sorted_sampler, self.batch_size, self.drop_last))):
                yield [bucket[i] for i in batch]

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return math.ceil(len(self.sampler) / self.batch_size)


def get_number_of_elements(object_):
    """ Get the sum of the number of elements in all tensors stored in `object_`.

    This is particularly useful for sampling the largest objects based on tensor size like in:
    `OomBatchSampler.__init__.get_item_size`.

    Args:
        object (any)

    Returns:
        (int): The number of elements in the `object_`.
    """
    return sum([t.numel() for t in get_tensors(object_)])


class OomBatchSampler(BatchSampler):
    """ Out-of-memory (OOM) batch sampler wraps `batch_sampler` to sample the `num_batches` largest
    batches first in attempt to cause any potential OOM errors early.

    Credits:
    https://github.com/allenai/allennlp/blob/3d100d31cc8d87efcf95c0b8d162bfce55c64926/allennlp/data/iterators/bucket_iterator.py#L43

    Args:
        batch_sampler (torch.utils.data.sampler.BatchSampler)
        get_item_size (callable): Measure the size of an item given it's index `int`.
        num_batches (int, optional): The number of the large batches to move to the beginning of the
            iteration.
    """

    def __init__(self, batch_sampler, get_item_size, num_batches=5):
        self.batch_sampler = batch_sampler
        self.get_item_size = get_item_size
        self.num_batches = num_batches

    def __iter__(self):
        batches = list(iter(self.batch_sampler))
        largest_batches = heapq.nlargest(
            self.num_batches,
            range(len(batches)),
            key=lambda i: sum([self.get_item_size(j) for j in batches[i]]))
        move_to_front = [batches[i] for i in largest_batches]
        [batches.pop(i) for i in sorted(largest_batches, reverse=True)]
        batches[0:0] = move_to_front
        return iter(batches)

    def __len__(self):
        return len(self.batch_sampler)


class DistributedBatchSampler(BatchSampler):
    """ `BatchSampler` wrapper that distributes across each batch multiple workers.

    Arguments:
        batch_sampler (torch.utils.data.sampler.BatchSampler)
        num_replicas (int, optional): Number of processes participating in distributed training.
        rank (int, optional): Rank of the current process within num_replicas.
    """

    def __init__(self, batch_sampler, num_replicas=None, rank=None):
        if not src.distributed.is_initialized():
            raise RuntimeError("Requires distributed to be initialized.")

        self.batch_sampler = batch_sampler
        self.num_replicas = (
            torch.distributed.get_world_size() if num_replicas is None else num_replicas)
        self.rank = torch.distributed.get_rank() if rank is None else rank

    def __iter__(self):
        for batch in self.batch_sampler:
            yield [e for i, e in enumerate(batch) if (i - self.rank) % self.num_replicas == 0]

    def __len__(self):
        return len(self.batch_sampler)


class RepeatSampler(Sampler):
    """ Sampler that repeats forever.

    Args:
        sampler (torch.data.utils.sampler.Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


class DeterministicSampler(Sampler):
    """ `DeterministicSampler` forks the random state for another sampler that it wraps.

    Args:
        sampler (torch.data.utils.sampler.Sampler)
        random_seed (int, optional): Uses a `get_initial_seed()` if not defined.
        cuda (bool, optional): If `True` this sampler forks the random state of CUDA as well.
    """

    def __init__(self, sampler, random_seed=None, cuda=False):
        self.sampler = sampler
        self.rng_state = None
        self.random_seed = get_initial_seed() if random_seed is None else random_seed
        self.cuda = cuda

    @contextmanager
    def _fork_rng(self):
        with fork_rng(cuda=self.cuda):
            if self.rng_state is not None:
                set_random_generator_state(self.rng_state)
            else:
                set_seed(self.random_seed)
            yield
            self.rng_state = get_random_generator_state(cuda=self.cuda)

    def __iter__(self):
        with self._fork_rng():
            iterator = iter(self.sampler)

        while True:
            try:
                with self._fork_rng():
                    sample = next(iterator)
                yield sample
            except StopIteration:
                break

    def __len__(self):
        return len(self.sampler)

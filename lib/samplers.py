# type: ignore
import math
import typing

from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torchnlp.samplers.sorted_sampler import SortedSampler

from lib.utils import identity


class BucketBatchSampler(BatchSampler):
    """`BucketBatchSampler` similar to `torchnlp`. This version incrementally creates the bucket,
    instead of all at once.

    Original:
    https://github.com/PetrochukM/PyTorch-NLP/blob/master/torchnlp/samplers/bucket_batch_sampler.py
    """

    def __init__(
        self,
        sampler,
        batch_size,
        drop_last,
        sort_key: typing.Callable = identity,
        bucket_size_multiplier=100
    ):
        super().__init__(sampler, batch_size, drop_last)
        self.sort_key = sort_key
        self.bucket_size = batch_size * bucket_size_multiplier
        if hasattr(sampler, "__len__"):
            self.bucket_size = min(self.bucket_size, len(sampler))
        self.batch_sampler = BatchSampler(sampler, batch_size, False)

    def __iter__(self):
        sampler = iter(self.sampler)
        bucket = [next(sampler) for _ in range(self.bucket_size)]
        while len(bucket) > 0:
            sorted_sampler = SortedSampler(bucket, self.sort_key)
            bucket = []
            batches = list(BatchSampler(sorted_sampler, self.batch_size, self.drop_last))
            for batch in SubsetRandomSampler(batches):
                yield [sorted_sampler.data[i] for i in batch]
                bucket.extend(s for _, s in zip(range(len(batch)), sampler))

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return math.ceil(len(self.sampler) / self.batch_size)

import logging
import time

from torch.multiprocessing import cpu_count
from torchnlp.samplers import RepeatSampler
from tqdm import tqdm

import torch
import torch.utils.data

from src.environment import IS_TESTING_ENVIRONMENT
from src.utils.utils import identity
from src.utils.utils import log_runtime
from src.utils.utils import seconds_to_string

logger = logging.getLogger(__name__)


class _DataLoaderDataset(torch.utils.data.Dataset):
    """ Dataset that allows for a callable upon loading a single example.

    Args:
        dataset (torch.utils.data. Dataset): Dataset from which to load the data.
        load_fn (callable): Function to run on `__getitem__`.
    """

    def __init__(self, dataset, load_fn):
        self.dataset = dataset
        self.load_fn = load_fn

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.load_fn(self.dataset[index])


class DataLoader(torch.utils.data.dataloader.DataLoader):
    """ PyTorch DataLoader that supports a ``load_fn``.

    Args:
        dataset (torch.utils.data. Dataset): Dataset from which to load the data.
        load_fn (callable, optional): Callable run to load a single row of the dataset.
        post_process_fn (callable, optional): Callable run directly before the batch is returned.
        num_workers (int, optional): Number of subprocesses to use for data loading. Given a 0 value
          the data will be loaded in the main process.
        use_tqdm (bool, optional): Log a TQDM progress bar.
        **kwargs: Other key word arguments to be passed to ``torch.utils.data.DataLoader``
    """

    @log_runtime
    def __init__(self,
                 dataset,
                 load_fn=identity,
                 post_processing_fn=identity,
                 num_workers=0 if IS_TESTING_ENVIRONMENT else min(cpu_count(), 6),
                 use_tqdm=False,
                 **kwargs):
        super().__init__(
            dataset=_DataLoaderDataset(dataset, load_fn), num_workers=num_workers, **kwargs)

        # NOTE: At the moment, zero batches will cause an infinite loop.
        assert len(self.batch_sampler) > 0, 'There must be at least one batch.'

        logger.info('Launching with %d workers', self.num_workers)
        self.post_processing_fn = post_processing_fn
        # LEARN MORE: https://github.com/pytorch/pytorch/issues/15849#issuecomment-557668847
        object.__setattr__(self, 'batch_sampler', RepeatSampler(self.batch_sampler))
        self.use_tqdm = use_tqdm
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        start = time.time()
        iterator = range(len(self))
        iterator = tqdm(iterator) if self.use_tqdm else iterator
        for i in iterator:
            batch = self.post_processing_fn(next(self.iterator))
            if i == 0:
                elapsed = seconds_to_string(time.time() - start)
                logger.info('Time to first batch was %s.', elapsed)
            yield batch

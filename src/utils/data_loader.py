import logging
import time

from torch.multiprocessing import cpu_count
from tqdm import tqdm

import torch
import torch.utils.data

from src.environment import IS_TESTING_ENVIRONMENT
from src.utils.utils import identity
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
        trial_run (bool, optional): If ``True`` iteration stops after the first batch and doesn't
          start any workers.
        use_tqdm (bool, optional): Log a TQDM progress bar.
        **kwargs: Other key word arguments to be passed to ``torch.utils.data.DataLoader``
    """

    def __init__(self,
                 dataset,
                 load_fn=identity,
                 post_processing_fn=identity,
                 num_workers=0 if IS_TESTING_ENVIRONMENT else cpu_count(),
                 trial_run=False,
                 use_tqdm=False,
                 **kwargs):
        super().__init__(
            dataset=_DataLoaderDataset(dataset, load_fn), num_workers=num_workers, **kwargs)

        self.trial_run = trial_run
        if self.trial_run:
            self.num_workers = 0

        logger.info('Launching with %d workers', self.num_workers)
        self.post_processing_fn = post_processing_fn
        self.use_tqdm = use_tqdm

    def __len__(self):
        return 1 if self.trial_run else super().__len__()

    def __iter__(self):
        start = time.time()
        is_first = True

        iterator = super().__iter__()
        if self.use_tqdm:
            iterator = tqdm(iterator, total=len(self))

        for batch in iterator:
            batch_ = self.post_processing_fn(batch)

            if is_first:
                elapsed = seconds_to_string(time.time() - start)
                logger.info('Time to first batch was %s.', elapsed)
                is_first = False

            yield batch_

            if self.trial_run:
                break

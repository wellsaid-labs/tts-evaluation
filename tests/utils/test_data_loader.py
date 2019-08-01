from collections import namedtuple

from torchnlp.utils import collate_tensors

import torch

from src.utils.data_loader import _DataLoaderDataset
from src.utils.data_loader import DataLoader


def test_data_loader_dataset():
    expected = [2, 3]
    dataset = _DataLoaderDataset([1, 2], lambda x: x + 1)
    assert len(dataset) == len(expected)
    assert list(dataset) == expected


def test_data_loader():
    dataset = [1]
    for batch in DataLoader(
            dataset,
            trial_run=True,
            post_processing_fn=lambda x: x + 1,
            load_fn=lambda x: x + 1,
            use_tqdm=True):
        assert len(batch) == 1
        assert batch[0] == 3


MockTuple = namedtuple('MockTuple', ['t'])


def test_data_loader__named_tuple__collate_fn():
    dataset = [MockTuple(t=torch.Tensor(1)), MockTuple(t=torch.Tensor(1))]
    for batch in DataLoader(dataset, batch_size=2, collate_fn=collate_tensors):
        assert batch.t.shape == (2, 1)

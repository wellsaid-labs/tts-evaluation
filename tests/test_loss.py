import os

from torch.nn import MSELoss

import torch

from src.loss import plot_loss
from src.loss import Loss


def test_plot_loss():
    filename = 'tests/_test_data/loss.png'
    plot_loss([[4, 3, 2, 1], [3, 2, 1, 0]], ['Train', 'Valid'], filename)

    assert os.path.isfile(filename)
    # Clean up
    os.remove(filename)


def test_loss():
    criterion = Loss(MSELoss)
    input_ = torch.FloatTensor([1, 2, 3])
    target = torch.FloatTensor([3, 2, 1])
    loss = criterion(input_, target)
    assert loss == 8 / 3


def test_loss_reduce():
    criterion = Loss(MSELoss, reduce=False)
    input_ = torch.FloatTensor([1, 2, 3])
    target = torch.FloatTensor([3, 2, 1])
    loss = criterion(input_, target)
    assert torch.equal(loss, torch.FloatTensor([4, 0, 4]))


def test_loss_mask():
    criterion = Loss(MSELoss)
    input_ = torch.FloatTensor([1, 2, 3])
    target = torch.FloatTensor([3, 2, 1])
    mask = torch.FloatTensor([1, 1, 0])
    loss = criterion(input_, target, mask=mask)
    assert loss == 4 / 2


def test_loss_epoch():
    criterion = Loss(MSELoss)

    # Step one
    input_ = torch.FloatTensor([1, 2, 3])
    target = torch.FloatTensor([3, 2, 1])
    mask = torch.FloatTensor([1, 1, 0])
    criterion(input_, target, mask=mask)

    # Step two
    input_ = torch.FloatTensor([1, 2, 3])
    target = torch.FloatTensor([3, 2, 1])
    criterion(input_, target)

    assert criterion.epoch() == (8 + 4) / (2 + 3)


def test_loss_get_attr():
    criterion = Loss(MSELoss)
    assert hasattr(criterion, 'cuda')

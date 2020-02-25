import math

from unittest import mock

import torch

from src.utils import DistributedAveragedMetric
from src.utils import AveragedMetric


def test_averaged_metric():
    metric = AveragedMetric()
    metric.update(torch.tensor([.5]), torch.tensor(3))
    metric.update(0.25, 2)

    assert metric.last_update() == 0.25
    assert metric.reset() == 0.4
    assert math.isnan(metric.last_update())


@mock.patch('torch.distributed')
def test_distributed_average_metric(mock_distributed):
    mock_distributed.reduce.return_value = None
    mock_distributed.is_initialized.return_value = True

    metric = DistributedAveragedMetric()
    metric.update(torch.tensor([.5]), torch.tensor(3))
    metric.update(0.25, 2)

    assert metric.last_update() == 0.25
    assert metric.sync().last_update() == 0.4
    assert metric.reset() == 0.4
    assert math.isnan(metric.last_update())


def test_distributed_average_metric__not_initialized():
    metric = DistributedAveragedMetric()
    metric.update(torch.tensor([.5]), torch.tensor(3))
    metric.update(0.25, 2)

    assert metric.last_update() == 0.25
    assert metric.sync().last_update() == 0.4
    assert metric.reset() == 0.4
    assert math.isnan(metric.last_update())

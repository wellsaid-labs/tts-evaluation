from unittest import mock

import torch

from src.utils import AccumulatedMetrics


@mock.patch('torch.distributed')
def test_accumulated_metrics(mock_distributed):
    mock_distributed.reduce.return_value = None
    mock_distributed.is_initialized.return_value = True
    metrics = AccumulatedMetrics(type_=torch)
    metrics.add_metric('test', torch.tensor([.5]), torch.tensor(3))
    metrics.add_metrics({'test': torch.tensor([.25])}, 2)

    def callable_(key, value):
        assert key == 'test' and value == 0.4

    metrics.log_step_end(callable_)
    assert metrics.get_epoch_metric('test') == 0.4
    metrics.log_epoch_end(callable_)
    metrics.reset()

    called = False

    def not_called():
        nonlocal called
        called = True

    # No new metrics to report
    metrics.log_step_end(not_called)
    metrics.log_epoch_end(not_called)

    assert not called

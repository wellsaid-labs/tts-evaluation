from collections import defaultdict

import torch
import torch.utils.data

import src.distributed


class AccumulatedMetrics():
    """
    Args:
        type_: Default torch tensor type.
    """

    def __init__(self, type_=torch.cuda):
        self.type_ = type_
        self.reset()

    def reset(self):
        self.metrics = {
            'epoch_total': defaultdict(float),
            'epoch_count': defaultdict(float),
            'step_total': defaultdict(float),
            'step_count': defaultdict(float)
        }

    def add_metric(self, name, value, count=1):
        """ Add metric as part of the current step.

        Args:
            name (str)
            value (number)
            count (int): Number of times to add value / frequency of value.
        """
        if torch.is_tensor(value):
            value = value.item()

        if torch.is_tensor(count):
            count = count.item()

        assert count > 0, '%s count, %f, must be a positive number' % (name, count)

        self.metrics['step_total'][name] += value * count
        self.metrics['step_count'][name] += count

    def add_metrics(self, dict_, count=1):
        """ Add multiple metrics as part of the current step.

        Args:
            dict_ (dict): Metrics in the form of key value pairs.
            count (int): Number of times to add value / frequency of value.
        """
        for metric, value in dict_.items():
            self.add_metric(metric, value, count)

    def log_step_end(self, log_metric):
        """ Log all metrics that have been accumulated since the last ``log_step_end``.

        Note that in the distributed case, only the master node gets the accurate metric.

        Args:
            log_metric (callable(key, value)): Callable to log a metric.
        """
        # Temporary until this issue is fixed: https://github.com/pytorch/pytorch/issues/20651
        if len(self.metrics['step_total']) == 0 and len(self.metrics['step_count']) == 0:
            return

        # Accumulate metrics between multiple processes.
        if src.distributed.is_initialized():
            metrics_total_items = sorted(self.metrics['step_total'].items(), key=lambda t: t[0])
            metrics_total_values = [value for _, value in metrics_total_items]

            metrics_count_items = sorted(self.metrics['step_count'].items(), key=lambda t: t[0])
            metrics_count_values = [value for _, value in metrics_count_items]

            packed = self.type_.FloatTensor(metrics_total_values + metrics_count_values)
            torch.distributed.reduce(packed, dst=src.distributed.get_master_rank())
            packed = packed.tolist()

            for (key, _), value in zip(metrics_total_items, packed[:len(metrics_total_items)]):
                self.metrics['step_total'][key] = value

            for (key, _), value in zip(metrics_count_items, packed[len(metrics_total_items):]):
                self.metrics['step_count'][key] = value

        # Log step metrics and update epoch metrics.
        for (total_key, total_value), (count_key, count_value) in zip(
                self.metrics['step_total'].items(), self.metrics['step_count'].items()):

            assert total_key == count_key, 'AccumulatedMetrics invariant failed.'
            assert count_value > 0, 'AccumulatedMetrics invariant failed (%s, %f, %f)'
            log_metric(total_key, total_value / count_value)

            self.metrics['epoch_total'][total_key] += total_value
            self.metrics['epoch_count'][total_key] += count_value

        # Reset step metrics
        self.metrics['step_total'] = defaultdict(float)
        self.metrics['step_count'] = defaultdict(float)

    def get_epoch_metric(self, metric):
        """ Get the current epoch value for a metric.
        """
        return self.metrics['epoch_total'][metric] / self.metrics['epoch_count'][metric]

    def log_epoch_end(self, log_metric):
        """ Log all metrics that have been accumulated since the last ``log_epoch_end``.

        Args:
            log_metric (callable(key, value)): Callable to log a metric.
        """
        self.log_step_end(lambda *args, **kwargs: None)

        # Log epoch metrics
        for (total_key, total_value), (count_key, count_value) in zip(
                self.metrics['epoch_total'].items(), self.metrics['epoch_count'].items()):

            assert total_key == count_key, 'AccumulatedMetrics invariant failed.'
            assert count_value > 0, 'AccumulatedMetrics invariant failed (%s, %f, %f)'
            log_metric(total_key, total_value / count_value)

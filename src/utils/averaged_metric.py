import logging

import torch
import torch.utils.data

import src.distributed

logger = logging.getLogger(__name__)


class AveragedMetric():
    """ Compute and track the average of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """ Reset the metric statistics.

        Returns:
            (float): The average value before reseting.
        """
        if hasattr(self, 'total_value') and hasattr(self, 'total_count') and self.total_count > 0:
            average = self.total_value / self.total_count
        else:
            average = None

        self.last_update_value = 0
        self.last_update_count = 0
        self.total_value = 0
        self.total_count = 0

        return average

    def update(self, value, count=1):
        """ Add a new measurement of this metric.

        Args:
            value (number)
            count (int): Number of times to add value / frequency of value.

        Returns:
            AverageMetric: `self`
        """
        if torch.is_tensor(value):
            value = value.item()

        if torch.is_tensor(count):
            count = count.item()

        assert count > 0, '%s count must be a positive number' % count

        self.total_value += value * count
        self.total_count += count
        self.last_update_value = value * count
        self.last_update_count = count

        return self

    def last_update(self):
        """ Get the measurement update.

        Returns:
            (float): The average value of the last measurement.
        """
        if self.last_update_count == 0:
            return None

        return self.last_update_value / self.last_update_count


class DistributedAveragedMetric(AveragedMetric):
    """ Compute and track the average of a metric in a distributed environment.
    """

    def __init__(self):
        super().__init__()

    def reset(self):
        super().reset()

        if (hasattr(self, 'post_sync_total_value') and hasattr(self, 'post_sync_total_value') and
                self.post_sync_total_count > 0):
            average = self.post_sync_total_value / self.post_sync_total_count
        else:
            average = None

        self.post_sync_total_value = 0
        self.post_sync_total_count = 0

        return average

    def sync(self):
        """ Synchronize measurements from multiple processes.

        Returns:
            AverageMetric: `self`
        """
        last_post_sync_total_value = self.post_sync_total_value
        last_post_sync_total_count = self.post_sync_total_count
        if src.distributed.is_initialized():
            torch_ = torch.cuda if torch.cuda.is_available() else torch
            packed = torch_.FloatTensor([self.total_value, self.total_count])
            torch.distributed.reduce(packed, dst=src.distributed.get_master_rank())
            self.post_sync_total_value, self.post_sync_total_count = tuple(packed.tolist())
        else:
            self.post_sync_total_value = self.total_value
            self.post_sync_total_count = self.total_count
        self.last_update_value = self.post_sync_total_value - last_post_sync_total_value
        self.last_update_count = self.post_sync_total_count - last_post_sync_total_count
        return self

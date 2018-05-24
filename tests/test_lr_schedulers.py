from unittest import TestCase

from torch.optim import SGD

import torch
import torch.nn.functional as F

from src.lr_schedulers import DelayedExponentialLR

# PORTED FROM:
# https://github.com/pytorch/pytorch/blob/e44f901b55873ebb6b1b0d3bab30fd89d487b71c/test/test_optim.py


class SchedulerTestNet(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 1, 1)
        self.conv2 = torch.nn.Conv2d(1, 1, 1)

    def forward(self, x):
        return self.conv2(F.relu(self.conv1(x)))


class TestLRScheduler(TestCase):

    def setUp(self):
        self.net = SchedulerTestNet()
        self.optim = SGD([{'params': self.net.conv1.parameters()}], lr=10e-3)

    def test_delayed_exponential_lr(self):
        epoch_end_decay = 60000
        epoch_start_decay = 10000
        end_lr = 10e-5
        epochs = 70000
        step = 10000
        targets = [[10e-3, 10e-3, 0.003981, 0.001584, 0.00063, 0.00025, 10e-5, 10e-5]]
        scheduler = DelayedExponentialLR(
            self.optim,
            epoch_end_decay=epoch_end_decay,
            epoch_start_decay=epoch_start_decay,
            end_lr=end_lr)

        for epoch in range(epochs):
            scheduler.step(epoch)
            if epoch % step == 0:
                for param_group, target in zip(self.optim.param_groups, targets):
                    self.assertAlmostEqual(
                        target[epoch // step],
                        param_group['lr'],
                        msg='LR is wrong in epoch {}: expected {}, got {}'.format(
                            epoch, target[epoch // step], param_group['lr']),
                        delta=1e-5)

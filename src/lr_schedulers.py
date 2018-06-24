import math
import logging

from torch.optim.lr_scheduler import _LRScheduler

from src.utils.configurable import configurable

logger = logging.getLogger(__name__)


class DelayedExponentialLR(_LRScheduler):
    """Set the learning rate of each parameter group to the initial lr decayed
    by gamma every epoch. When last_epoch=-1, sets initial lr as lr.

    SOURCE (Tacotron 2):
        learning rate of 10−3 exponentially decaying to 10−5 starting after 50,000 iterations.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        epoch_end_decay (int): Epoch to end exponential decay
        end_lr (float): Learning rate to be reached at ``epoch_end_decay``
        epoch_start_decay (int): Epoch to start exponential decay
        last_epoch (int): The index of last epoch. Default: -1.
    """

    @configurable
    def __init__(self, optimizer, epoch_end_decay, end_lr, epoch_start_decay=0, last_epoch=-1):
        self.epoch_start_decay = epoch_start_decay

        # Create a scheduler when ``last_epoch!= -1``
        # https://github.com/moskomule/senet.pytorch/blob/c3fe7e6d8916c57024d13fdcf489bc5f94119180/utils.py#L76
        for group in optimizer.param_groups:
            group.setdefault('initial_lr', group['lr'])

        super().__init__(optimizer, last_epoch)

        self.gammas = [
            math.exp(math.log(end_lr / base_lr) / (epoch_end_decay - epoch_start_decay))
            for base_lr in self.base_lrs
        ]
        self.epoch_end_decay = epoch_end_decay
        self.end_lr = end_lr

    def get_lr(self):
        if self.last_epoch <= self.epoch_start_decay:
            return self.base_lrs
        elif self.last_epoch <= self.epoch_end_decay:
            return [
                base_lr * gamma**(self.last_epoch - self.epoch_start_decay)
                for gamma, base_lr in zip(self.gammas, self.base_lrs)
            ]

        return [self.end_lr for _ in self.base_lrs]

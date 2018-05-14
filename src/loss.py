import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_loss(losses, names, filename, title='Loss', timestep='Epoch'):
    """ Plot loss over time.

    Args:
        losses (list of lists): Lost of loss datapoints per step
        names (list): Name of each loss curve
        filename (str): Location to save the file.
        title (str, optional): Title of the plot.
        timestep (str, optional): Name of the x axis representing a timestep (i.e. epoch)
    """
    assert '.png' in filename.lower(), "Filename saves in PNG format"
    plt.style.use('ggplot')
    colors = matplotlib.cm.tab10.colors
    for i, loss in enumerate(losses):
        color = colors[i % len(colors)]
        plt.plot(
            list(range(len(loss))),
            loss,
            color=color,
            marker='.',
            linestyle='solid',
            label=names[i])
        plt.plot(np.argmin(loss), min(loss), color=color, marker='v')
        plt.plot(np.argmax(loss), max(loss), color=color, marker='^')
    plt.title(title, y=1.1)
    plt.ylabel('Loss')
    plt.xlabel(timestep)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(filename, format='png')
    plt.close()


# TODO: Replace this with Tensorboard


class Loss(object):
    """ Loss object that keeps track of average loss over time for an epoch.

    Args:
        criterion (callable): ``torch.nn.modules.loss._Loss`` criterion to instantiate.
        *args: List of arguments used to passed to criterion.
        size_average (bool, optional): By default, the losses are averaged over
            observations for each minibatch. However, if the field :attr:`size_average`
            is set to ``False``, the losses are instead summed for each minibatch.
            Default: ``True``
        reduce (bool, optional): By default, the losses are averaged or summed over
            observations for each minibatch depending on :attr:`size_average`. When
            :attr:`reduce` is ``False``, returns a loss per batch element instead and
            ignores :attr:`size_average`. Default: ``True``
        *kwargs: List of keyword arguments used to passed to criterion.
    """

    def __init__(self, criterion, *args, size_average=True, reduce=True, **kwargs):
        self.criterion = criterion(*args, reduce=False)
        self.size_average = size_average
        self.reduce = reduce
        self.total = 0
        self.num_values = 0

    def __call__(self, input_, target, *args, mask=None, **kwargs):
        if mask is None:
            values = int(np.prod(input_.shape))
        else:
            mask = mask.expand_as(target)
            values = torch.sum(mask).item()
        self.num_values += values

        loss = self.criterion(input_, target, *args, **kwargs)
        if mask is not None:
            loss = loss * mask

        sum_ = torch.sum(loss)
        self.total += sum_.item()
        if not self.reduce:
            return loss

        return sum_ / values if self.size_average else sum_

    def epoch(self):
        """ Complete an epoch with loss function, reseting the loss statistics for the next epoch.

        Returns:
            (float) Average loss over the epoch
        """
        epoch_loss = self.total / self.num_values
        self.total = 0
        self.num_values = 0
        return epoch_loss

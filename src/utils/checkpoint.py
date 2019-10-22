from pathlib import Path

import glob
import logging
import os

from torchnlp.random import get_random_generator_state

import torch
import torch.utils.data

from src.utils.utils import flatten_parameters
from src.utils.utils import load
from src.utils.utils import save

logger = logging.getLogger(__name__)


class Checkpoint():
    """ Torch model checkpoint object.

    Args:
        directory (Path or str): Directory where to save the checkpoint.
        model (torch.nn.Module): Model to train and evaluate.
        step (int): Starting step, useful warm starts (i.e. checkpoints).
        **kwargs (dict, optional): Any other checkpoint attributes.
    """

    def __init__(self, directory, step, model, **kwargs):
        self.directory = Path(directory)
        self.step = step
        self.model = model
        self.path = Path(self.directory) / 'step_{}.pt'.format(self.step)

        # TODO: Consider using the `NamedTuple` approach for attribute naming with underscores. The
        # approach allows attributes to be populated by the user but also allows having some
        # built-in attributes.
        # Learn more:
        # https://softwareengineering.stackexchange.com/questions/315348/why-is-the-replace-method-of-python-namedtuple-classes-protected
        for key, value in kwargs.items():
            assert not hasattr(self, key), 'This checkpoint attribute already exists.'
            setattr(self, key, value)

    @classmethod
    def from_path(class_, path, device=torch.device('cpu')):
        """ Load a checkpoint from path.

        NOTE: Given ``path`` is different than the loaded instance, the original path is not
        overwritten.

        Args:
            path (Path or str): Path to a checkpoint to load.
            device (torch.device, optional): Device to load checkpoint onto.

        Returns:
            checkpoint (Checkpoint): Loaded checkpoint.
        """
        instance = load(str(path), device=device)
        flatten_parameters(instance.model)
        instance.path = Path(path)
        logger.info('Loaded checkpoint at step %d from %s with model:\n%s', instance.step,
                    instance.path, instance.model)
        return instance

    @classmethod
    def most_recent(class_, pattern, **kwargs):
        """ Load the most recent checkpoint from ``root``.

        Args:
            pattern (str): Pattern to glob recursively for checkpoints.
            **kwargs (dict, optional): Any additional parameters to pass to ``class_.from_path``.

        Returns:
            (Checkpoint or None): The most recent checkpoint found or None if none is found.
        """
        checkpoints = list(glob.iglob(str(pattern), recursive=True))
        if len(checkpoints) == 0:
            logger.warning('No checkpoints found in %s' % pattern)
            return None

        checkpoints = sorted(list(checkpoints), key=os.path.getctime, reverse=True)
        for path in checkpoints:
            try:
                return class_.from_path(path, **kwargs)
            except (EOFError, RuntimeError):
                logger.exception('Failed to load checkpoint %s' % path)
                pass

        raise ValueError('Unable to load recent checkpoint.')

    def save(self):
        """ Save a checkpoint. """
        self.random_generator_state = get_random_generator_state()
        save(self.path, self)
        return self.path

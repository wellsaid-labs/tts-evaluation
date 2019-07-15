from collections import namedtuple
from pathlib import Path

import glob
import logging
import os
import random

import numpy as np
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

        for key, value in kwargs.items():
            setattr(self, key, value)

    @classmethod
    def from_path(class_, path, device=torch.device('cpu'), load_random_state=True):
        """ Load a checkpoint from path.

        NOTE: Given ``path`` is different than the loaded instance, the original path is not
        overwritten.

        Args:
            path (Path or str): Path to a checkpoint to load.
            device (torch.device, optional): Device to load checkpoint onto.
            load_random_state (bool, optional): Load the random state with the checkpoint.

        Returns:
            checkpoint (Checkpoint): Loaded checkpoint.
        """
        from src.datasets import Gender  # NOTE: Prevent circular dependency
        from src.datasets import Speaker

        instance = load(str(path), device=device)
        if hasattr(instance, 'random_generator_state'):
            if load_random_state:
                set_random_generator_state(instance.random_generator_state)
        else:
            logger.warning('Old Checkpoint: unable to load checkpoint random generator state')

        if (hasattr(instance, 'input_encoder') and Speaker(
                'Frank Bonacquisti', Gender.FEMALE) in instance.input_encoder.speaker_encoder.stoi):
            logger.warning('Old Checkpoint: detected and fixed Frank\'s gender')

            old_speaker = Speaker('Frank Bonacquisti', Gender.FEMALE)
            new_speaker = Speaker('Frank Bonacquisti', Gender.MALE)

            index = instance.input_encoder.speaker_encoder.stoi[old_speaker]
            instance.input_encoder.speaker_encoder.itos[index] = new_speaker
            del instance.input_encoder.speaker_encoder.stoi[old_speaker]
            instance.input_encoder.speaker_encoder.stoi[new_speaker] = index

            count = instance.input_encoder.speaker_encoder.tokens[old_speaker]
            del instance.input_encoder.speaker_encoder.tokens[old_speaker]
            instance.input_encoder.speaker_encoder.tokens[old_speaker] = count

        flatten_parameters(instance.model)
        # Backwards compatibility for instances without paths.
        instance.path = instance.path if hasattr(instance, 'path') else path
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


RandomGeneratorState = namedtuple('RandomGeneratorState',
                                  ['random', 'torch', 'numpy', 'torch_cuda'])


def get_random_generator_state():
    """ Get the `torch`, `numpy` and `random` random generator state.

    Returns:
        RandomGeneratorState
    """
    return RandomGeneratorState(
        random.getstate(), torch.random.get_rng_state(), np.random.get_state(),
        torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def set_random_generator_state(state):
    """ Set the `torch`, `numpy` and `random` random generator state.

    Args:
        state (RandomGeneratorState)
    """
    logger.info('Setting the random state for `torch`, `numpy` and `random`.')
    random.setstate(state.random)
    torch.random.set_rng_state(state.torch)
    np.random.set_state(state.numpy)
    if torch.cuda.is_available() and state.torch_cuda is not None and len(
            state.torch_cuda) == torch.cuda.device_count():
        torch.cuda.set_rng_state_all(state.torch_cuda)

import argparse
import dill
import logging
import os
import platform
import random
import subprocess
import sys
import time

import numpy as np
import tensorflow as tf
import torch

from src.configurable import log_config
# from src.configurable import log_arguments
from src.hparams import set_hparams
from src.utils import get_root_path


class CopyStream(object):
    """ Wrapper that copies a stream to a file without affecting the stream.

    **Reference:** https://stackoverflow.com/questions/4675728/redirect-stdout-to-a-file-in-python

    Example:
        >>> sys.stdout = CopyStream(stdout_filename, sys.stdout)
    """

    def __init__(self, filename, stream):
        self.stream = stream
        self.file_ = open(filename, 'a')

    @property
    def closed(self):
        return self.file_.closed and self.stream.closed

    def __getattr__(self, attr):
        # Run the functions ``flush``, ``close``, and ``write`` for both ``self.stream`` and
        # ``self.file``.
        if attr in ['flush', 'close', 'write']:
            return lambda *args, **kwargs: (getattr(self.file_, attr)(*args, **kwargs) and
                                            getattr(self.stream, attr)(*args, **kwargs))
        return getattr(self.stream, attr)


class ExperimentContextManager(object):
    """ ExperimentContextManager is a context manager for PyTorch enabling reproducible experiments.

    Side Effect:
        In order to redirect stdout, resets the logging module and setups new handlers.

    Args:
        label (str): Label, typically the model name, for a set of experiments.
        directory (str): Top level directory for all experiments.
    """

    def __init__(self, label='', top_level_directory='experiments/', seed=1212212, device=None):
        self.label = label
        self.root_path = get_root_path()
        self.top_level_directory = os.path.join(self.root_path, top_level_directory)

        if device is None:
            self.device = torch.cuda.current_device() if torch.cuda.is_available() else -1
        else:
            self.device = device
        self.is_cuda = self.device >= 0
        self.set_seed(seed)

    def load(self, path):
        """ Using ``torch.load`` and ``dill`` load an object from ``path`` onto ``self.device``.

        Args:
            path (str): Filename to load in ``self.directory``

        Returns:
            (any): Object loaded.
        """

        def remap(storage, loc):
            if 'cuda' in loc and self.device >= 0:
                return storage.cuda(device=self.device)
            return storage

        return torch.load(
            os.path.join(self.directory, path), map_location=remap, pickle_module=dill)

    def save(self, path, data):
        """ Using ``torch.save`` and ``dill`` save an object to ``path`` in ``self.directory``

        Args:
            path (str): Filename to save to in ``self.directory``
            data (any): Data to save into file.

        Returns:
            (str): Full path saved too
        """
        full_path = os.path.join(self.directory, path)
        torch.save(data, full_path, pickle_module=dill)
        return full_path

    def notify(self, title, text):
        """ Queue a desktop notification on a Linux or OSX machine.

        Args:
            title (str): Notification title
            text (str): Notification description
        """
        system = platform.system()
        cmd = ''
        if system == 'Darwin':  # OSX
            cmd = """osascript -e 'display notification "{text}" with title "{title}"'"""
        elif system == 'Linux':
            cmd = "notify-send '{title}' '{text}'"

        os.system(cmd.format(text=text, title=title))

    def usage(self):
        """ Print CUDA usage """
        os.system('nvidia-smi')

    def _copy_standard_streams(self, stdout_filename='stdout.log', stderr_filename='stderr.log'):
        """ Copy stdout and stderr to a ``{directory}/stdout.log`` and ``{directory}/stderr.log``.

        Args:
            directory (str): Directory to save streams.
            stdout_filename (str): Filename used to save the stdout stream.
            stderr_filename (str): Filename used to save the stderr stream.
        """
        self.stdout_filename = os.path.join(self.directory, stdout_filename)
        self.stderr_filename = os.path.join(self.directory, stderr_filename)
        sys.stdout = CopyStream(self.stdout_filename, sys.stdout)
        sys.stderr = CopyStream(self.stderr_filename, sys.stderr)

    def _new_experiment_directory(self):
        """ Create a experiment directory named with ``self.label`` and the current time.

        Returns:
            path (str): Path to the new experiment directory
        """
        name = '%s.%s' % (self.label, time.strftime('%m_%d_%H:%M:%S', time.localtime()))
        path = os.path.join(self.top_level_directory, name)
        os.makedirs(path)
        return path

    def git_commit(self):
        """ Commit the current code structure before every experiment with an emoji: 🎓

        Args:
            messsage (str): Message to add to git commit
        """
        subprocess.call([
            'git', 'commit', '-am',
            ':mortar_board: %s | %s' % (os.path.basename(self.directory), self.message)
        ])

    def set_seed(self, seed):
        """ To ensure reproducibility, by seeding ``numpy``, ``random``, ``tf`` and ``torch``.

        Args:
            seed (int): Integer used as seed.
        """
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        tf.set_random_seed(seed)
        if self.is_cuda:
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        self.seed = seed

    def maybe_cuda(self, any_, **kwargs):
        """ Move ``any`` to CUDA iff ``self.is_cuda`` is True.

        Args:
            any_ (any): Any object that has ``cuda`` attribute.
            **kwargs (dict): Other keyword arguments to pass to ``cuda`` callable.

        Returns:
            (any): Object after calling the ``cuda`` callable.
        """
        return any_.cuda(device=self.device, **kwargs) if self.is_cuda else any_

    def __enter__(self):
        """ Runs before the experiment context begins.
        """
        # LEARN MORE:
        # https://stackoverflow.com/questions/42270739/how-do-i-resolve-these-tensorflow-warnings
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

        # Create a local directory to store logs, checkpoints, etc..
        self.directory = self._new_experiment_directory()

        # Copy ``stdout`` and ``stderr`` to experiments folder
        self._copy_standard_streams()

        # Setup logging
        self._stream_handler = logging.StreamHandler(stream=sys.stdout)
        self._stream_handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '[%(asctime)s][%(processName)s][%(name)s][%(levelname)s] %(message)s')
        self._stream_handler.setFormatter(formatter)
        root = logging.getLogger()
        root.addHandler(self._stream_handler)
        root.setLevel(logging.INFO)

        logger = logging.getLogger(__name__)
        logger.info('Experiment Folder: %s' % self.directory)
        logger.info('Label: %s', self.label)
        logger.info('Device: %d', self.device)
        logger.info('Seed: %s', self.seed)

        parser = argparse.ArgumentParser()
        parser.add_argument('message', help='Message describing the experiment.')
        self.message = parser.parse_args().message

        # Set the hyperparameters with configurable
        set_hparams()

        # Log the hyperparameter configuration
        log_config()

        return self

    def __exit__(self, type_, value, traceback):
        """ Runs after the experiment context ends.
        """
        # Print all arguments used
        # log_arguments()

        # Flush stdout and stderr to capture everything
        sys.stdout.flush()
        sys.stderr.flush()

        # Reset streams
        sys.stdout = sys.stdout.stream
        sys.stderr = sys.stderr.stream

        logging.getLogger().removeHandler(self._stream_handler)

        if type_ is None:  # No exception
            self.git_commit()

        self.notify('Experiment: %s' % os.path.basename(self.directory), 'Experiment has exited.')

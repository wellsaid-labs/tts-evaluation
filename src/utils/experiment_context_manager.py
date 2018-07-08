import logging
import os
import platform
import random
import sys
import time
import shutil

import numpy as np
import torch

from src.utils.visualize import Tensorboard

logger = logging.getLogger(__name__)


class _CopyStream(object):
    """ Wrapper that copies a stream to a file without affecting the stream.

    **Reference:** https://stackoverflow.com/questions/4675728/redirect-stdout-to-a-file-in-python

    Example:
        >>> sys.stdout = _CopyStream(stdout_filename, sys.stdout)
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
    """ Context manager for seeding, organizing and recording experiments.

    Args:
        directory (str, optional): Directory to save experiment in.
        label (str, optional): Group a set of experiments with a label, typically the model name.
        name (str, optional): Name of the experiment.
        root (str, optional): Top level directory for all experiments.
        seed (int, optional): The seed to use.
        device (torch.Device, optional): Set a device. By default, we the device is:
            ``torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')``
        min_time (int, optional): If an experiment is less than ``min_time`` in seconds, then it's
            files are deleted.
    """

    def __init__(self,
                 directory=None,
                 label='other',
                 name=None,
                 root='experiments/',
                 seed=1212212,
                 device=None,
                 min_time=60 * 3,
                 step=0):
        # Handle circular reference
        from src.utils import ROOT_PATH

        self.directory = directory
        self.id = str(time.time())[:10]  # NOTE: Same id tensorboard uses.
        self.name = time.strftime('%H:%M:%S', time.localtime()) if name is None else name
        self.label = label
        self.root = os.path.normpath(os.path.join(ROOT_PATH, root))

        self.device = device
        if self.device is None:
            self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.is_cuda = self.device.type == 'cuda'
        self.set_seed(seed)
        self.min_time = min_time
        self._start_time = time.time()
        self.step = step

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

        logger.info('Notify: %s', text)

        os.system(cmd.format(text=text, title=title))

    def _copy_standard_streams(self, stdout_filename='stdout.log', stderr_filename='stderr.log'):
        """ Copy stdout and stderr to a ``{directory}/stdout.log`` and ``{directory}/stderr.log``.

        Args:
            directory (str): Directory to save streams.
            stdout_filename (str): Filename used to save the stdout stream.
            stderr_filename (str): Filename used to save the stderr stream.
        """
        self.stdout_filename = os.path.join(self.directory, '%s.%s' % (self.id, stdout_filename))
        self.stderr_filename = os.path.join(self.directory, '%s.%s' % (self.id, stderr_filename))
        sys.stdout = _CopyStream(self.stdout_filename, sys.stdout)
        sys.stderr = _CopyStream(self.stderr_filename, sys.stderr)

    def _new_experiment_directory(self):
        """ Create a experiment directory with checkpoints.

        Returns:
            path (str): Path to the new experiment directory
        """
        if self.directory is None:
            run_day = time.strftime('%m_%d', time.localtime())
            name = self.name.replace(' ', '_')
            self.directory = os.path.join(self.root, self.label, run_day, name)
            assert not os.path.isdir(
                self.directory), 'Attempting to override an existing experiment %s' % self.directory
            os.makedirs(self.directory)
        else:
            assert os.path.isdir(self.directory), 'Provided directory must exist.'

        # Make checkpoints directory
        self.checkpoints_directory = os.path.join(self.directory, 'checkpoints', self.id)
        os.makedirs(self.checkpoints_directory)

    def _tensorboard(self):
        """ Within ``self.directory`` setup tensorboard.
        """
        # NOTE: ``tb`` short name to help with Tensorboard path layout in the left hand
        log_dir = os.path.join(self.directory, 'tb')
        self.dev_tensorboard = Tensorboard(log_dir=os.path.join(log_dir, 'dev'), step=self.step)
        self.train_tensorboard = Tensorboard(log_dir=os.path.join(log_dir, 'train'), step=self.step)

    def set_seed(self, seed):
        """ To ensure reproducibility, by seeding ``numpy``, ``random``, ``tf`` and ``torch``.

        Args:
            seed (int): Integer used as seed.
        """
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        if self.is_cuda:
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        self.seed = seed

    def __enter__(self):
        """ Runs before the experiment context begins.
        """
        # Create a local directory to store logs, checkpoints, etc..
        self._new_experiment_directory()

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

        if self.is_cuda and self.device.index is not None:
            torch.cuda.set_device(self.device.index)

        logger = logging.getLogger(__name__)
        logger.info('Experiment Folder: %s' % self.directory)
        logger.info('Label: %s', self.label)
        logger.info('Device: %s', self.device)
        logger.info('Seed: %s', self.seed)
        logger.info('Step: %d', self.step)
        logger.info('ID: %s', self.id)

        self._tensorboard()

        return self

    def clean_up(self):
        """ Delete files associated with this context. """
        logger.info('Deleting Experiment: %s', self.directory)
        shutil.rmtree(self.directory)

        # Remove empty directories
        for root, directories, files in os.walk(self.root, topdown=False):
            for directory in directories:
                directory = os.path.join(root, directory)
                # Only works when the directory is empty
                try:
                    os.rmdir(directory)
                except OSError:
                    pass

    def __exit__(self, exception, value, traceback):
        """ Runs after the experiment context ends.
        """
        self.dev_tensorboard.close()
        self.train_tensorboard.close()

        # NOTE: Log before removing handlers.
        elapsed_seconds = time.time() - self._start_time
        if self.min_time is not None and elapsed_seconds < self.min_time and exception:
            self.clean_up()

        self.notify('Experiment', 'Experiment has exited after %d seconds.' % (elapsed_seconds))

        # Flush stdout and stderr to capture everything
        sys.stdout.flush()
        sys.stderr.flush()

        # Reset streams
        sys.stdout = sys.stdout.stream
        sys.stderr = sys.stderr.stream

        logging.getLogger().removeHandler(self._stream_handler)

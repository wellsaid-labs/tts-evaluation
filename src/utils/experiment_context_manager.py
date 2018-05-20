import logging
import os
import platform
import random
import sys
import time
import shutil

import numpy as np
import tensorflow as tf
import torch
from tensorboardX import SummaryWriter

from src.utils.configurable import log_config
from src.utils.configurable import log_arguments

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
        label (str, optional): Group a set of experiments with a label, typically the model name.
        root (str, optional): Top level directory for all experiments.
        seed (int, optional): The seed to use.
        device (torch.Device, optional): Set a device. By default, we the device is:
            ``torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')``
        min_time (int, optional): If an experiment is less than ``min_time`` in seconds, then it's
            files are deleted.
    """

    def __init__(self, label='other', root='experiments/', seed=1212212, device=None, min_time=60):
        # Fix circular reference
        from src.utils import ROOT_PATH

        self.label = label
        self.root = os.path.normpath(os.path.join(ROOT_PATH, root))

        self.device = device
        if self.device is None:
            self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.is_cuda = self.device.type == 'cuda'
        self.set_seed(seed)
        self.min_time = min_time
        self._start_time = time.time()

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
        self.stdout_filename = os.path.join(self.directory, stdout_filename)
        self.stderr_filename = os.path.join(self.directory, stderr_filename)
        sys.stdout = _CopyStream(self.stdout_filename, sys.stdout)
        sys.stderr = _CopyStream(self.stderr_filename, sys.stderr)

    def _new_experiment_directory(self):
        """ Create a experiment directory with epochs, tensorboard and time organization.

        Returns:
            path (str): Path to the new experiment directory
        """
        run_day = time.strftime('%m_%d', time.localtime())
        run_time = time.strftime('%H:%M:%S', time.localtime())
        self.directory = os.path.join(self.root, self.label, run_day, run_time)
        os.makedirs(self.directory)

        # Make checkpoints directory
        self.checkpoints_directory = os.path.join(self.directory, 'checkpoints')
        os.makedirs(self.checkpoints_directory)

        # Setup tensorboard
        self.tensorboard = SummaryWriter(log_dir=os.path.join(self.directory, 'tensorboard'))

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

    def __enter__(self):
        """ Runs before the experiment context begins.
        """
        # Fix a circular reference chain
        from src.hparams import set_hparams

        # LEARN MORE:
        # https://stackoverflow.com/questions/42270739/how-do-i-resolve-these-tensorflow-warnings
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

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

        logger = logging.getLogger(__name__)
        logger.info('Experiment Folder: %s' % self.directory)
        logger.info('Label: %s', self.label)
        logger.info('Device: %s', self.device)
        logger.info('Seed: %s', self.seed)

        # Set the hyperparameters with configurable
        set_hparams()

        # Log the hyperparameter configuration
        log_config()

        return self

    def __exit__(self, type_, value, traceback):
        """ Runs after the experiment context ends.
        """
        # Print all arguments used
        log_arguments()

        # Flush stdout and stderr to capture everything
        sys.stdout.flush()
        sys.stderr.flush()

        # Reset streams
        sys.stdout = sys.stdout.stream
        sys.stderr = sys.stderr.stream

        logging.getLogger().removeHandler(self._stream_handler)

        elapsed_seconds = time.time() - self._start_time
        if self.min_time is not None and elapsed_seconds < self.min_time:
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

        self.notify('Experiment', 'Experiment has exited after %d seconds.' % (elapsed_seconds))

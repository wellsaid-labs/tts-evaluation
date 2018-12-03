from pathlib import Path

import logging
import os
import platform
import random
import sys
import time
import shutil
import subprocess

import numpy as np
import torch

from src.utils import duplicate_stream

import src.distributed

logger = logging.getLogger(__name__)


class TrainingContextManager(object):
    """ Context manager for seeding, organizing and recording training runs.

    Args:
        root_directory (str): The directory to store the run.
        seed (int, optional): The seed to use.
        device (torch.Device, optional): Set a default device. By default, we the device is:
            ``torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')``.
        min_time (int, optional): If an run is less than ``min_time`` in seconds, then it's
            files are deleted.
        step (int, optional): The step training is started at.
    """

    def __init__(self, root_directory, seed=1212212, device=None, min_time=60 * 15, step=0):
        self.root_directory = Path(root_directory)
        # NOTE: ``self.id`` is used to distinguish this run if started from a checkpoint in the
        # same ``root_directory```
        self.id = str(time.time())[:10]
        if torch.distributed.is_initialized():
            self.id = src.distributed.broadcast_string(self.id)
            if step == 0 and self.root_directory.is_dir() and src.distributed.is_master():
                raise ValueError('Directory path is already in use %s' % str(self.root_directory))
        else:  # CASE: Not distributed
            if step == 0 and self.root_directory.is_dir():
                raise ValueError('Directory path is already in use %s' % str(self.root_directory))

        if device is None:
            self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        else:
            self.device = device

        self.is_cuda = self.device.type == 'cuda'
        self.min_time = min_time
        self.start_time = time.time()
        self.seed = seed

    def _notify(self, title, text):
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

    def _check_module_versions(self):
        """ Check to that ``requirements.txt`` is in-line with ``pip freeze`` """
        freeze = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'])
        freeze = freeze.decode('utf-8').split()
        with Path('requirements.txt').open() as file_:
            for line in file_:
                line = line.strip()
                if '==' in line:
                    specification = line.split()[0]
                    package = specification.split('==')[0]
                    installed = [p for p in freeze if p.split('==')[0] == package]
                    if not len(installed) == 1:
                        raise ValueError('%s not installed' % package)
                    if not specification == installed[0]:
                        raise ValueError(
                            'Versions are not compatible %s =/= %s' % (specification, installed[0]))

    def _copy_standard_streams(self, stdout_filename='stdout.log', stderr_filename='stderr.log'):
        """ Copy stdout and stderr to a ``{directory}/stdout.log`` and ``{directory}/stderr.log``.

        Args:
            stdout_filename (str): Filename used to save the stdout stream.
            stderr_filename (str): Filename used to save the stderr stream.
        """
        self.stdout_filename = self.root_directory / ('%s.%s.%s' %
                                                      (self.id, self.device.index, stdout_filename))
        self.stderr_filename = self.root_directory / ('%s.%s.%s' %
                                                      (self.id, self.device.index, stderr_filename))
        self.stop_duplicate_stream_stdout = duplicate_stream(sys.stdout, self.stdout_filename)
        self.stop_duplicate_stream_stderr = duplicate_stream(sys.stderr, self.stderr_filename)

    def _set_seed(self, seed):
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
        # NOTE: Set first incase there are any random operations afterwards.
        self._set_seed(self.seed)
        self.checkpoints_directory = self.root_directory / 'checkpoints' / self.id

        if not torch.distributed.is_initialized() or src.distributed.is_master():
            self.root_directory.mkdir(parents=True, exist_ok=True)
            self.checkpoints_directory.mkdir(parents=True)

        # Must run before using distributed tensors
        if self.is_cuda and self.device.index is not None:
            torch.cuda.set_device(self.device.index)

        # Sync processes before using stdout and ``self.root_directory.mkdir`` ran.
        if torch.distributed.is_initialized():
            src.distributed.sync()

        # Setup logging
        self._copy_standard_streams()

        logger = logging.getLogger(__name__)
        logger.info('Experiment Folder: %s' % self.root_directory)
        logger.info('Device: %s', self.device)
        logger.info('Seed: %s', self.seed)
        logger.info('ID: %s', self.id)
        if torch.distributed.is_initialized():
            logger.info('World Size: %d' % torch.distributed.get_world_size())

        self._check_module_versions()

        return self

    def clean_up(self):
        """ Delete files associated with this context.
        """
        if torch.distributed.is_initialized() and not src.distributed.is_master():
            return

        logger.info('DELETING EXPERIMENT: %s', self.root_directory)

        # Remove checkpoints
        shutil.rmtree(str(self.checkpoints_directory))

        # Remove log files
        self.stdout_filename.unlink()
        self.stderr_filename.unlink()

        # Remove empty directories
        for root, directories, files in os.walk(str(self.root_directory), topdown=False):
            for directory in directories:
                directory = Path(root) / directory
                # NOTE: Only works when the directory is empty
                try:
                    directory.rmdir()
                except OSError:
                    pass

        # NOTE: Only works when the directory is empty
        try:
            self.root_directory.rmdir()
        except OSError:
            pass

    def __exit__(self, exception, value, traceback):
        """ Runs after the experiment context ends.
        """
        # Reset streams
        self.stop_duplicate_stream_stderr()
        self.stop_duplicate_stream_stdout()

        # NOTE: Log before removing handlers.
        elapsed_seconds = time.time() - self.start_time
        is_short_experiment = self.min_time is not None and elapsed_seconds < self.min_time
        if is_short_experiment and exception:
            self.clean_up()

        self._notify('Experiment', 'Experiment has exited after %d seconds.' % (elapsed_seconds))

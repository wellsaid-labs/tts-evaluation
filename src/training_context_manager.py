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

from src.utils.visualize import Tensorboard
from src.utils.utils import duplicate_stream

logger = logging.getLogger(__name__)


class ExperimentContextManager(object):
    """ Context manager for seeding, organizing and recording experiments.

    Args:
        directory (str, optional): Directory to save experiment in.
        label (str, optional): Group a set of experiments with a label, typically the model name.
            If ``directory`` is provided, this option is ignored.
        name (str, optional): Name of the experiment.
        root (str, optional): Top level directory for all experiments.
        seed (int, optional): The seed to use.
        device (torch.Device, optional): Set a device. By default, we the device is:
            ``torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')``
        min_time (int, optional): If an experiment is less than ``min_time`` in seconds, then it's
            files are deleted.
        step (int, optional): Step to record in creating tensorboard.
    """

    def __init__(self,
                 directory=None,
                 label=None,
                 name=None,
                 root='experiments/',
                 seed=1212212,
                 device=None,
                 min_time=60 * 3,
                 step=0):
        self.directory = None if directory is None else Path(directory)
        self.started_from_existing_directory = self.directory is not None
        self.id = str(time.time())[:10]  # NOTE: Same id tensorboard uses.
        self.name = time.strftime('%H:%M:%S', time.localtime()) if name is None else name
        self.label = label
        self.root = Path(root)

        self.device = device
        if self.device is None:
            self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.is_cuda = self.device.type == 'cuda'
        self.set_seed(seed)
        self.min_time = min_time
        self._start_time = time.time()
        self.step = step

    def notify(self, title, text, use_logger=True):
        """ Queue a desktop notification on a Linux or OSX machine.

        Args:
            title (str): Notification title
            text (str): Notification description
            use_logger (bool, optional): Print statements with logger or ``print``.
        """
        system = platform.system()
        cmd = ''
        if system == 'Darwin':  # OSX
            cmd = """osascript -e 'display notification "{text}" with title "{title}"'"""
        elif system == 'Linux':
            cmd = "notify-send '{title}' '{text}'"

        if use_logger:
            logger.info('Notify: %s', text)
        else:
            print('Notify: %s' % text)

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
        self.stdout_filename = self.directory / ('%s.%s' % (self.id, stdout_filename))
        self.stderr_filename = self.directory / ('%s.%s' % (self.id, stderr_filename))
        self.stop_duplicate_stream_stdout = duplicate_stream(sys.stdout, self.stdout_filename)
        self.stop_duplicate_stream_stderr = duplicate_stream(sys.stderr, self.stderr_filename)

    def _new_experiment_directory(self):
        """ Create a experiment directory with checkpoints.

        Returns:
            path (str): Path to the new experiment directory
        """
        if self.directory is None:
            run_day = time.strftime('%m_%d', time.localtime())
            name = self.name.replace(' ', '_')
            self.directory = self.root / self.label / run_day / name
            assert not self.directory.is_dir(
            ), 'Attempting to override an existing experiment %s' % self.directory
            self.directory.mkdir(parents=True)
        else:
            assert self.directory.is_dir(), 'Provided directory must exist.'

        # Make checkpoints directory
        self.checkpoints_directory = self.directory / 'checkpoints' / self.id
        self.checkpoints_directory.mkdir(parents=True)

    def _tensorboard(self):
        """ Within ``self.directory`` setup tensorboard.
        """
        # NOTE: ``tb`` short name to help with Tensorboard path layout in the left hand
        log_dir = self.directory / 'tb'
        self.dev_tensorboard = Tensorboard(log_dir=log_dir / 'dev', step=self.step)
        self.train_tensorboard = Tensorboard(log_dir=log_dir / 'train', step=self.step)

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
        self._check_module_versions()

        # Create a local directory to store logs, checkpoints, etc..
        self._new_experiment_directory()

        # Copy ``stdout`` and ``stderr`` to experiments folder
        self._copy_standard_streams()

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='[%(asctime)s][' + str(self.device) + '][%(name)s][%(levelname)s] %(message)s')

        if self.is_cuda and self.device.index is not None:
            torch.cuda.set_device(self.device.index)

        logger = logging.getLogger(__name__)
        logger.info('Experiment Folder: %s' % self.directory)
        logger.info('Device: %s', self.device)
        logger.info('Seed: %s', self.seed)
        logger.info('Step: %d', self.step)
        logger.info('ID: %s', self.id)

        self._tensorboard()

        return self

    def clean_up(self, use_logger=True):
        """ Delete files associated with this context.

        Args:
            use_logger (bool, optional): Print statements with logger or ``print``.
        """
        if use_logger:
            logger.info('DELETING EXPERIMENT: %s', self.directory)
        else:
            print('DELETING EXPERIMENT: %s' % self.directory)

        if not self.started_from_existing_directory:
            shutil.rmtree(str(self.directory))
        else:
            # Remove checkpoints
            shutil.rmtree(str(self.checkpoints_directory))

            # Remove tensorboard files
            Path(self.dev_tensorboard.writer.file_writer.event_writer._ev_writer.
                 _py_recordio_writer.path).unlink()
            Path(self.train_tensorboard.writer.file_writer.event_writer._ev_writer.
                 _py_recordio_writer.path).unlink()

            # Remove log files
            self.stdout_filename.unlink()
            self.stderr_filename.unlink()

        # Remove empty directories
        for root, directories, files in os.walk(str(self.root), topdown=False):
            for directory in directories:
                directory = Path(root) / directory
                # Only works when the directory is empty
                try:
                    directory.rmdir()
                except OSError:
                    pass

    def __exit__(self, exception, value, traceback):
        """ Runs after the experiment context ends.
        """
        self.dev_tensorboard.close()
        self.train_tensorboard.close()

        # Reset streams
        self.stop_duplicate_stream_stderr()
        self.stop_duplicate_stream_stdout()

        # NOTE: Log before removing handlers.
        elapsed_seconds = time.time() - self._start_time
        is_short_experiment = self.min_time is not None and elapsed_seconds < self.min_time
        if is_short_experiment and exception:
            self.clean_up(use_logger=False)

        self.notify(
            'Experiment',
            'Experiment has exited after %d seconds.' % (elapsed_seconds),
            use_logger=False)

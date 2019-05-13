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

from src.utils import assert_enough_disk_space
from src.utils import duplicate_stream
from src.utils import ROOT_PATH
from src.utils import set_basic_logging_config

import src.distributed


class TrainingContextManager(object):
    """ Context manger for managing the training environment.

    Manages:
        seed: This module sets seeds for reproduction.
        logs: This module captures logs.
        disk: This module allocates a location for training data storage.
        versions: This module ensures PyPi module version correctness.
        distributed: This module initiates distributed training.

    Args:
        seed (int, optional): The seed to use.
        device (torch.Device, optional): Set a default device. By default, we the device is:
          ``torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')``.
        min_time (int, optional): If an run is less than ``min_time`` in seconds, then it's
          files are deleted.
    """

    def __init__(self, seed=1212212, device=None, min_time=60 * 15):
        self.__runtime_context = False

        if device is None:
            self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        else:
            self.device = device

        self.is_cuda = self.device.type == 'cuda'
        self.min_time = min_time
        self.start_time = time.time()
        self.seed = seed

    def init_distributed(self, backend='nccl'):
        """ Initiate distributed environment.

        Learn more: https://pytorch.org/tutorials/intermediate/dist_tuto.html

        Args:
           backend (str, optional): Name of the distributed backend to use.
        """
        torch.distributed.init_process_group(backend=backend)
        if self.is_cuda:
            torch.cuda.set_device(self.device.index)

    def _copy_standard_streams(self,
                               directory,
                               stdout_filename='stdout.log',
                               stderr_filename='stderr.log'):
        """ Copy stdout and stderr to a ``{directory}/stdout.log`` and ``{directory}/stderr.log``.

        NOTE: ``_copy_standard_streams`` only captures logs one location at a time. Executing
        ``_copy_standard_streams`` twice, will transition from the old location to the
        new.

        Args:
            directory (Path): Avaiable directory to store ``stdout`` and ``stderr`` files.
            stdout_filename (str): Captured stdout stream filename.
            stderr_filename (str): Captured stderr stream filename.
        """
        if src.distributed.is_master():
            directory.mkdir(parents=True, exist_ok=True)
        if src.distributed.is_initialized():  # Ensure ``mkdir`` runs on master before usage.
            torch.distributed.barrier()

        stdout_filename = directory / stdout_filename
        stderr_filename = directory / stderr_filename

        # Clean up old ``_copy_standard_streams`` references.
        if hasattr(self, 'stop_duplicate_stream_stdout'):
            self.stop_duplicate_stream_stdout()
        if hasattr(self, 'stop_duplicate_stream_stderr'):  # Close streams
            self.stop_duplicate_stream_stderr()
        if hasattr(self, 'stdout_filename') and self.stdout_filename.exists():  # Move files
            self.stdout_filename.replace(stdout_filename)
        if hasattr(self, 'stderr_filename') and self.stderr_filename.exists():
            self.stderr_filename.replace(stderr_filename)

        self.stdout_filename = stdout_filename
        self.stderr_filename = stderr_filename
        self.stop_duplicate_stream_stdout = duplicate_stream(sys.stdout, self.stdout_filename)
        self.stop_duplicate_stream_stderr = duplicate_stream(sys.stderr, self.stderr_filename)

    def set_context_root(self, root_directory, at_checkpoint=False):
        """ Set the root directory for storing data on disk.

        Args:
            root_directory (str): Directory on disk to store training data.
            at_checkpoint (bool): `True` if root directory is at a previous checkpoint location.
        """
        if not self.__runtime_context:
            raise TypeError('Must be used inside the runtime context.')

        self.root_directory = Path(root_directory)
        if not at_checkpoint and self.root_directory.is_dir() and src.distributed.is_master():
            raise TypeError('Directory path is already in use %s' % str(self.root_directory))
        if src.distributed.is_master():
            self.root_directory.mkdir(parents=True, exist_ok=True)
        if src.distributed.is_initialized():  # Ensure ``mkdir`` runs on master before usage.
            torch.distributed.barrier()
        self.logger.info('Run Directory: %s', root_directory)

        assert_enough_disk_space()  # Ensure there is roughly enough disk space for data storage.

        # NOTE: ``self.id`` is used to distinguish this run from other runs in the same directory.
        self.id = str(int(round(time.time())))
        self.logger.info('ID: %s', self.id)
        if src.distributed.is_initialized():
            # Ensure every distributed node has the same ``self.id``.
            self.id = src.distributed.broadcast_string(self.id)

        self.checkpoints_directory = self.root_directory / 'checkpoints' / self.id
        if src.distributed.is_master():
            self.checkpoints_directory.mkdir(parents=True)

        self._copy_standard_streams(self.root_directory,
                                    '%s.%s.%s' % (self.id, self.device.index, 'stdout.log'),
                                    '%s.%s.%s' % (self.id, self.device.index, 'stderr.log'))

    def _check_module_versions(self):
        """ Ensure installed modules respect ``requirements.txt`` """
        freeze = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'])
        freeze = freeze.decode('utf-8').split()
        requirements = Path(ROOT_PATH / 'requirements.txt').read_text()
        for line in requirements.split():
            line = line.strip()
            if '==' in line:
                specification = line.split()[0]
                package = specification.split('==')[0]
                installed = [p for p in freeze if p.split('==')[0] == package]
                if not len(installed) == 1:
                    raise RuntimeError('%s not installed' % package)
                if not specification == installed[0]:
                    # NOTE: RuntimeError could cause ``Illegal seek`` while running PyTest.
                    raise RuntimeError(
                        'Versions are not compatible %s =/= %s' % (specification, installed[0]))

    def _set_seed(self, seed):
        """ Set seed values for random generators.

        Args:
            seed (int): Value used as a seed.
        """
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        if self.is_cuda:
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        self.seed = seed
        self.logger.info('Seed: %s', self.seed)

    def __enter__(self):
        self.__runtime_context = True  # Has entered the runtime context.

        # Setup logging with disk storage
        # TODO: Set default paths like ``tmp`` via ``hparams/``
        prefix = '%d.%s' % (int(round(time.time())), self.device.index)
        self._copy_standard_streams(ROOT_PATH / 'tmp', '%s.stdout.log' % prefix,
                                    '%s.stderr.log' % prefix)
        set_basic_logging_config(self.device)
        self.logger = logging.getLogger(__name__)

        # Log metadata
        self.logger.info('Device: %s', self.device)
        if src.distributed.is_initialized():
            self.logger.info('World Size: %d' % torch.distributed.get_world_size())

        self._set_seed(self.seed)
        self._check_module_versions()

        return self

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

        self.logger.info('Notify: %s', text)
        os.system(cmd.format(text=text, title=title))

    def clean_up(self):
        """ Clean up files associated with this context runtime.
        """
        if not src.distributed.is_master():
            return

        if hasattr(self, 'root_directory'):
            self.logger.info('DELETING EXPERIMENT: %s', self.root_directory)

        # Remove checkpoints
        if hasattr(self, 'checkpoints_directory'):
            shutil.rmtree(str(self.checkpoints_directory))

        # Remove log files
        if hasattr(self, 'stdout_filename'):
            self.stdout_filename.unlink()
        if hasattr(self, 'stderr_filename'):
            self.stderr_filename.unlink()

        # Remove empty directories
        if hasattr(self, 'root_directory'):
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
        self.__runtime_context = False

        # Reset streams
        if hasattr(self, 'stop_duplicate_stream_stderr'):
            self.stop_duplicate_stream_stderr()
        if hasattr(self, 'stop_duplicate_stream_stdout'):
            self.stop_duplicate_stream_stdout()

        # NOTE: Log before removing handlers.
        elapsed_seconds = time.time() - self.start_time
        is_short_experiment = self.min_time is not None and elapsed_seconds < self.min_time
        if is_short_experiment and exception:
            self.clean_up()

        self._notify('Experiment', 'Experiment has exited after %d seconds.' % (elapsed_seconds))

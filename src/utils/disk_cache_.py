from functools import partial
from functools import wraps
from pathlib import Path
from threading import RLock

import atexit
import gc
import inspect
import logging
import pickle
import weakref

from src.environment import IS_TESTING_ENVIRONMENT
from src.environment import ROOT_PATH
from src.environment import TTS_DISK_CACHE_NAME
from src.utils.utils import ResetableTimer

logger = logging.getLogger(__name__)

# NOTE: The file was named `disk_cache_.py` instead of `disk_cache.py` because:
# https://stackoverflow.com/questions/40509588/patch-function-with-same-name-as-module-python-django-mock


class _Cache(object):
    """ Cache the `args` and `kwargs` with the associated returned value.

    Args:
        function (callable)
    """
    _instances = []

    def __init__(self, function):
        self.__class__._instances.append(weakref.ref(self))
        self._co_varnames = function.__code__.co_varnames
        self._storage = {}
        self._write_lock_property = None

    @classmethod
    def get_instances(class_):
        resolved = [i() for i in class_._instances]
        return [i for i in resolved if i is not None]

    @property
    def _write_lock(self):
        # NOTE: Lazily create `RLock` because of these issues:
        # https://github.com/pytorch/pytorch/issues/23117
        if self._write_lock_property is None:
            self._write_lock_property = RLock()
        return self._write_lock_property

    def _get_key(self, args=(), kwargs={}):
        """
        TODO: Add support variable `*args`.

        Args:
            args (tuple): Arguments passed to `function`.
            kwargs (dict): Keyword arguments password to `function`.

        Returns:
            frozenset: Set compromising of both `args` and `kwargs`.
        """
        # Learn more: https://stackoverflow.com/questions/830937/python-convert-args-to-kwargs
        key = kwargs.copy()
        key.update(zip(self._co_varnames, args))
        return frozenset(key.items())

    def clear(self):
        with self._write_lock:
            self._storage = {}

    def set_(self, args=(), kwargs={}, return_=None):
        key = self._get_key(args=args, kwargs=kwargs)

        if key in self._storage:
            assert return_ == self._storage[key], 'Overriding values in cache is not permitted.'
            return

        with self._write_lock:
            self._storage[key] = return_

    def get(self, args=(), kwargs={}):
        return self._storage[self._get_key(args=args, kwargs=kwargs)]

    def __iter__(self):
        # NOTE: `iter` breaks if `self._storage` is mutated during iteration.
        with self._write_lock:
            return iter(self._storage)

    def update(self, other_cache):
        if self._co_varnames != other_cache._co_varnames:
            raise ValueError('`other_cache` must be instantiated to the same function.')

        with self._write_lock:
            self._storage.update(other_cache._storage)

    def __eq__(self, other):
        if not isinstance(other, _Cache):
            raise TypeError('Equality is not support between `_Cache` and `%s`' % type(other))

        return other._storage == self._storage

    def __len__(self):
        return len(self._storage)

    def __contains__(self, item):
        """
        Args:
            item (tuple): Tuple with `args` and `kwargs`.

        Returns:
            bool: `True` if `args` and `kwargs` pair is in `self._storage`.
        """
        return self._get_key(*item) in self._storage


class _DiskCache(_Cache):
    """ `_Cache` object supports saving and loading from disk.

    Args:
        function (callable): Function to decorate.
        directory (str or Path): Directory to save function cache.
        save_to_disk_delay (int): Following some delay (in seconds) between function
            calls save cache to disk.

    Returns:
        (callable)
    """

    def __init__(self, function, save_to_disk_delay, directory):
        super().__init__(function)

        self._file_name = inspect.getmodule(function).__name__ + '.' + function.__qualname__
        self._file_path = Path(directory / self._file_name)
        self._write_timer = None
        self.save_to_disk_delay = save_to_disk_delay
        self._storage_property = None
        atexit.register(self._atexit)

    def _atexit(self):
        if self._write_timer is not None:
            self._write_timer.cancel()

        if not IS_TESTING_ENVIRONMENT:
            self.save()

    def clear(self):
        """ Clear cache from disk and memory."""
        with self._write_lock:
            if self._file_path.exists():
                self._file_path.unlink()
            self._storage = {}

    def load(self):
        """ Load cache from disk into memory. """
        if self._storage_property is None:
            self._storage_property = {}

        if self._file_path.exists():
            with self._write_lock:  # TODO: Test write lock with various race conditions.
                bytes_ = self._file_path.read_bytes()

                # NOTE: Speed up `pickle.loads` via `gc`, learn more:
                # https://stackoverflow.com/questions/26860051/how-to-reduce-the-time-taken-to-load-a-pickle-file-in-python
                gc.disable()
                disk_storage = pickle.loads(bytes_)
                gc.enable()

                logger.info('Loaded `_DiskCache` of size %d for function `%s`.', len(disk_storage),
                            self._file_name)

                self._storage_property.update(disk_storage)

    def save(self):
        """ Save cache to disk. """
        if self._write_timer is not None:
            self._write_timer.cancel()

        # NOTE Ensure that while writing that the `disk_cache` is not updated; therefore, the
        # `disk_cache` will not lose any data on update.
        with self._write_lock:
            self.load()  # NOTE: Disk may contain items not in `self._storage`
            logger.info('Saving `_DiskCache` of size %d for function `%s`.', len(self),
                        self._file_name)
            new_disk_storage = pickle.dumps(self._storage)
            self._file_path.parent.mkdir(parents=True, exist_ok=True)
            self._file_path.write_bytes(new_disk_storage)

    def set_(self, *args, **kwargs):
        if self.save_to_disk_delay is not None:
            if self._write_timer is not None and self._write_timer.is_alive():
                self._write_timer.reset()
            else:
                self._write_timer = ResetableTimer(self.save_to_disk_delay, self.save)
                self._write_timer.start()

        return super().set_(*args, **kwargs)

    @property
    def _storage(self):
        """ `self._storage` with a lazy load from disk for saving cache'd data. """
        if self._storage_property is None:
            self.load()
        return self._storage_property

    @_storage.setter
    def _storage(self, value):
        self._storage_property = value


def disk_cache(
        function=None,
        directory=(ROOT_PATH / 'tests' / '_test_data' if IS_TESTING_ENVIRONMENT else ROOT_PATH) /
        TTS_DISK_CACHE_NAME / 'disk_cache',
        save_to_disk_delay=None if IS_TESTING_ENVIRONMENT else 180):
    """ Function decorator that caches all function calls and saves them to disk.

    Attrs:
        cache (_DiskCache): The function cache.

    Args:
        function (callable): Function to decorate.
        directory (str or Path, optional): Directory to save function cache.
        save_to_disk_delay (int, optional): Following some delay (in seconds) between function
            calls save cache to disk.

    Returns:
        (callable)
    """
    if not function:
        return partial(disk_cache, directory=directory, save_to_disk_delay=save_to_disk_delay)

    cache = _DiskCache(function, save_to_disk_delay=save_to_disk_delay, directory=directory)

    @wraps(function)
    def decorator(*args, **kwargs):
        if (args, kwargs) not in cache:
            cache.set_(args=args, kwargs=kwargs, return_=function(*args, **kwargs))
        return cache.get(args=args, kwargs=kwargs)

    decorator.cache = cache
    return decorator

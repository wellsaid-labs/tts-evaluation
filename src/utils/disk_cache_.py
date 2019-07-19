from functools import partial
from functools import wraps
from pathlib import Path
from threading import RLock

import inspect
import logging
import pickle

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

    def __init__(self, function):
        self._co_varnames = function.__code__.co_varnames
        self._storage = {}
        self._write_lock = RLock()  # Lock to halt any cache writes.

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
        with self._write_lock:
            assert key not in self._storage, 'Overriding values in cache is not permitted.'
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
        directory (str or Path, optional): Directory to save function cache.
        save_to_disk_delay (int, optional): Following some delay (in seconds) between function
            calls save cache to disk.

    Returns:
        (callable)
    """

    def __init__(self,
                 function,
                 save_to_disk_delay=60,
                 directory=ROOT_PATH / TTS_DISK_CACHE_NAME / '_DiskCache'):
        super().__init__(function)

        self._file_name = inspect.getmodule(function).__name__ + '.' + function.__qualname__
        self._file_path = Path(directory / self._file_name)
        self._write_timer = None
        self.save_to_disk_delay = save_to_disk_delay

        self.load()


    def load(self):
        """ Load cache from disk. """
        if self._file_path.exists():
            with self._write_lock:
                disk_storage = pickle.loads(self._file_path.read_bytes())
                logger.info('Loaded %d `_DiskCache` for function `%s`.' % (len(disk_storage),
                                                                           self._file_name))
                self._storage.update(disk_storage)

    def save(self):
        """ Save cache to disk. """
        print('save')
        # NOTE Ensure that while writing that the `disk_cache` is not updated; therefore, the
        # `disk_cache` will not lose any data on update.
        with self._write_lock:
            self.load()  # NOTE: Disk may contain items not in `self._storage`
            logger.info('Saving %d `_DiskCache` for function `%s`.' % (len(self), self._file_name))
            new_disk_storage = pickle.dumps(self._storage)
            self._file_path.parent.mkdir(parents=True, exist_ok=True)
            self._file_path.write_bytes(new_disk_storage)

    def set_(self, *args, **kwargs):
        print('_DiskCache.set_')
        if self._write_timer is not None and self._write_timer.is_alive():
            self._write_timer.reset()
        else:
            self._write_timer = ResetableTimer(self.save_to_disk_delay, self.save)
            self._write_timer.start()

        return super().set_(*args, **kwargs)


def disk_cache(function=None,
               directory=ROOT_PATH / TTS_DISK_CACHE_NAME / 'disk_cache',
               save_to_disk_delay=60):
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

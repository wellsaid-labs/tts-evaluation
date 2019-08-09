from functools import partial
from functools import wraps
from pathlib import Path
from threading import RLock

import atexit
import gc
import inspect
import logging
import os
import pickle
import tempfile
import weakref

from src.environment import DEFAULT_TTS_DISK_CACHE
from src.environment import IS_TESTING_ENVIRONMENT
from src.environment import TEMP_PATH
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
        self._lazy_write_lock = None

    @classmethod
    def get_instances(class_):
        """ Get all instances of `_Cache` objects.
        """
        resolved = [i() for i in class_._instances]
        return [i for i in resolved if i is not None]

    @property
    def _write_lock(self):
        # NOTE: Lazily create `RLock` because of these issues:
        # https://github.com/pytorch/pytorch/issues/23117
        if self._lazy_write_lock is None:
            self._lazy_write_lock = RLock()
        return self._lazy_write_lock

    def _get_key(self, args=(), kwargs={}):
        """ Get a unique key for `args` and `kwargs` various combinations.

        TODO: Add support variable `*args`.

        Learn more about the implementation:
        https://stackoverflow.com/questions/830937/python-convert-args-to-kwargs

        Args:
            args (tuple or None): Arguments passed to `function`.
            kwargs (dict or None): Keyword arguments password to `function`.

        Returns:
            frozenset: Set compromising of both `args` and `kwargs`.
        """
        if not isinstance(kwargs, dict) and kwargs is not None:
            raise TypeError('`kwargs` must be of type `dict` or `None`')

        if not isinstance(args, tuple) and args is not None:
            raise TypeError('`args` must be of type `tuple` or `None`')

        # NOTE: `copy` to avoid changing the provided `kwargs`.
        key = {} if kwargs is None else kwargs.copy()
        if args is not None:
            key.update(zip(self._co_varnames, args))
        return frozenset(key.items())

    def purge(self):
        """ Invalidate the cache.

        Learn more about a cache purge: https://en.wikipedia.org/wiki/Cache_invalidation
        """
        with self._write_lock:
            self._storage = {}

    def _set(self, key, return_):
        """ Private `_set` function ensuring that nothing is overwritten.

        Args:
            key (any): Arguments passed to `function`.
            return_ (any): The returned value of `function`.
        """
        if key in self._storage:
            if return_ != self._storage[key]:
                raise ValueError('Overriding values in a cache is not permitted.')
            return

        with self._write_lock:
            self._storage[key] = return_

    def set(self, args=(), kwargs={}, return_=None):
        """ Set `return_` given the parameters `args` and `kwargs.

        Args:
            args (tuple or None): Arguments passed to `function`.
            kwargs (dict or None): Keyword arguments password to `function`.
            return_ (any): The returned value of `function`.
        """
        self._set(self._get_key(args=args, kwargs=kwargs), return_)

    def get(self, args=(), kwargs={}):
        """ Get the `function` return value given `args` and `kwargs`.

        Raises:
            KeyError: Throws a key error if `args` and `kwargs` are not in cache.

        Args:
            args (tuple or None): Arguments passed to `function`.
            kwargs (dict or None): Keyword arguments password to `function`.

        Returns:
            any: The returned value of `function`.
        """
        return self._storage[self._get_key(args=args, kwargs=kwargs)]

    def __iter__(self):
        """ Iteratate over all cache values.
        """
        # NOTE: `iter` breaks if `self._storage` is mutated during iteration.
        with self._write_lock:
            yield from iter(self._storage)

    def update(self, other_cache):
        """ Update `self` with the contents of `other_cache`.

        Args:
            other_cache (Cache_): Another cache for the same function.
        """
        if self._co_varnames != other_cache._co_varnames:
            raise ValueError('`other_cache` must be instantiated to the same function.')

        for key in iter(other_cache):
            self._set(key, other_cache._storage[key])

    def __eq__(self, other_cache):
        """ Check if two `caches` are equal. """
        if not isinstance(other_cache, _Cache):
            raise TypeError('Equality is not support between `_Cache` and `%s`' % type(other_cache))

        if self._co_varnames != other_cache._co_varnames:
            return False

        return other_cache._storage == self._storage

    def __len__(self):
        """ Get the number of elements stored in the cache. """
        return len(self._storage)

    def contains(self, args=(), kwargs={}):
        """ Check if the cache contains `args` with `kwargs`.

        Args:
            args (tuple or None): Arguments passed to `function`.
            kwargs (dict or None): Keyword arguments password to `function`.

        Returns:
            bool: `True` if `args` and `kwargs` pair is in `self._storage`.
        """
        return self._get_key(args=args, kwargs=kwargs) in self._storage

    def __contains__(self, item):
        """ Check if the cache contains `args` with `kwargs`.

        Args:
            item (tuple or dict)

        Returns:
            bool: `True` if `args` and `kwargs` pair is in `self._storage`.
        """
        if isinstance(item, tuple):
            return self._get_key(args=item) in self._storage
        elif isinstance(item, dict):
            return self._get_key(kwargs=item) in self._storage

        raise TypeError('`item` must be either a `tuple` or `dict`.')


class _DiskCache(_Cache):
    """ `_Cache` object supports saving and loading from disk.

    Args:
        function (callable): Function to decorate.
        directory (str or Path): Directory to save function cache.
        save_to_disk_delay (int): Following some delay (in seconds) between function
            calls save cache to disk.
        get_file_name (callable, optional): Set the disk filename given the function.

    Returns:
        (callable)
    """

    def __init__(self,
                 function,
                 save_to_disk_delay,
                 directory,
                 get_file_name=lambda f: inspect.getmodule(f).__name__ + '.' + f.__qualname__):
        super().__init__(function)

        self._file_name = get_file_name(function)
        self._file_path = Path(directory / self._file_name)
        self._write_timer = None
        self.save_to_disk_delay = save_to_disk_delay
        self._lazy_storage = None

        # NOTE: `_disk_modified_date` is a floating point number giving the number of seconds since
        # the epoch.
        self._last_known_disk_storage_modified_date = -1.0
        self._lazy_disk_cache_size = None

        atexit.register(self._atexit)

    def _atexit(self):  # pragma: no cover
        """ Handler for program exit.
        """
        if self._write_timer is not None:
            self._write_timer.cancel()

        if not IS_TESTING_ENVIRONMENT:
            self.save()

    def _save_disk_metadata(self, disk_storage_modified_date, disk_cache_size):
        """ Save the current state of disk storage to memory. """
        self._lazy_disk_cache_size = disk_cache_size
        self._last_known_disk_storage_modified_date = disk_storage_modified_date

    @property
    def _storage(self):
        """ `self._storage` with a lazy load from disk for saving cache'd data. """
        if self._lazy_storage is None:
            self.load()

        return self._lazy_storage

    @_storage.setter
    def _storage(self, value):
        self._lazy_storage = value

    def load(self):
        """ Load cache from disk into memory.

        TODO: Test write lock with various race conditions.
        """
        if self._lazy_storage is None:
            self._lazy_storage = {}

        if self._file_path.exists():
            # NOTE: `str(self._file_path)` for Python 3.5 support.
            _disk_modified_date = os.path.getmtime(str(self._file_path))
            if self._last_known_disk_storage_modified_date < _disk_modified_date:
                with self._write_lock:
                    bytes_ = self._file_path.read_bytes()
                    # NOTE: Speed up `pickle.loads` via `gc`, learn more:
                    # https://stackoverflow.com/questions/26860051/how-to-reduce-the-time-taken-to-load-a-pickle-file-in-python
                    gc.disable()
                    disk_storage = pickle.loads(bytes_)
                    gc.enable()

                    logger.info('Loaded `_DiskCache` of size %d for function `%s`.',
                                len(disk_storage), self._file_name)

                    self._lazy_storage.update(disk_storage)

                    # NOTE: Only change disk metadata once we're confident we've
                    # ingested the data and won't need to load it again.
                    self._save_disk_metadata(_disk_modified_date, len(disk_storage))

    def _write_timer_save(self, *args, **kwargs):
        """ Alias for `save` useful for debugging via tracebacks. """
        return self.save(*args, **kwargs)

    def save(self):
        """ Save cache to disk. """
        if self._lazy_storage is None:
            return

        if self._write_timer is not None:
            self._write_timer.cancel()

        # NOTE: `with self._write_lock` ensures that while writing that the `disk_cache` it is
        # not updated by another thread.
        with self._write_lock:
            # NOTE: This may lose data with multiple processes: after `self.load()` and before
            # the save is complete another process may have written more data to disk.
            # NOTE: Disk may contain items not in `self._storage`.
            self.load()

            # NOTE: Since `_Cache` doesn't allow overrides or deletes, `self` can only be larger
            # or the same.
            if not self._file_path.exists() or len(self) > self._lazy_disk_cache_size:
                logger.info('Saving `_DiskCache` of size %d for function `%s`.', len(self),
                            self._file_name)
                new_disk_storage = pickle.dumps(self._storage)
                self._file_path.parent.mkdir(parents=True, exist_ok=True)

                # Atomic write, learn more:
                # http://stupidpythonideas.blogspot.com/2014/07/getting-atomic-writes-right.html
                # https://stackoverflow.com/questions/2333872/atomic-writing-to-file-with-python
                with tempfile.NamedTemporaryFile(dir=TEMP_PATH, delete=False, mode='wb') as file_:
                    file_.write(new_disk_storage)
                os.replace(file_.name, str(self._file_path))

                self._save_disk_metadata(os.path.getmtime(str(self._file_path)), len(self))

    def set(self, args=(), kwargs={}, return_=None):
        if self.save_to_disk_delay is not None and not self.contains(args=args, kwargs=kwargs):
            if self._write_timer is not None and self._write_timer.is_alive():
                self._write_timer.reset()
            else:
                self._write_timer = ResetableTimer(self.save_to_disk_delay, self._write_timer_save)
                self._write_timer.start()

        return super().set(args=args, kwargs=kwargs, return_=return_)

    def purge(self):
        with self._write_lock:
            if self._write_timer is not None:
                self._write_timer.cancel()

            if self._file_path.exists():
                self._file_path.unlink()
                self._save_disk_metadata(-1.0, 0)

            if self._lazy_storage is not None:
                super().purge()


def disk_cache(function=None,
               directory=DEFAULT_TTS_DISK_CACHE / 'disk_cache',
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
        if not cache.contains(args=args, kwargs=kwargs):
            cache.set(args=args, kwargs=kwargs, return_=function(*args, **kwargs))
        return cache.get(args=args, kwargs=kwargs)

    decorator.cache = cache
    return decorator

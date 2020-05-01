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

from src.environment import DISK_CACHE_PATH
from src.environment import IS_TESTING_ENVIRONMENT
from src.environment import TEMP_PATH

import src

logger = logging.getLogger(__name__)

# NOTE: The file was named `disk_cache_.py` instead of `disk_cache.py` because:
# https://stackoverflow.com/questions/40509588/patch-function-with-same-name-as-module-python-django-mock


class _Cache(object):
    """ Thread-safe key-value pair cache.
    """
    _instances = []

    def __init__(self):
        self.__class__._instances.append(weakref.ref(self))
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

    def purge(self):
        """ Invalidate the cache.

        Learn more about a cache purge: https://en.wikipedia.org/wiki/Cache_invalidation
        """
        with self._write_lock:
            self._storage = {}

    def set(self, key, value):
        if key in self._storage:
            if value != self._storage[key]:
                raise ValueError('Overriding value `%s` with `%s` for `%s` is not permitted.' %
                                 (self._storage[key], value, key))
            return

        with self._write_lock:
            self._storage[key] = value

    def get(self, key):
        return self._storage[key]

    def __iter__(self):
        # NOTE: `iter` breaks if `self._storage` is mutated during iteration.
        with self._write_lock:
            yield from iter(self._storage)

    def update(self, other_cache):
        for key in iter(other_cache):
            self.set(key, other_cache._storage[key])

    def __eq__(self, other_cache):
        if not isinstance(other_cache, _Cache):
            raise TypeError('Equality is not support between `_Cache` and `%s`' % type(other_cache))

        return other_cache._storage == self._storage

    def __len__(self):
        return len(self._storage)

    def __contains__(self, key):
        return key in self._storage


class DiskCache(_Cache):
    """ `_Cache` that supports saving and loading from disk.

    Args:
        filename (str or Path): Location to save the disk cache.
        save_to_disk_delay (int): This saves to disk after every lookup, after a delay (in seconds).

    Returns:
        (callable)
    """

    def __init__(self, filename, save_to_disk_delay=None if IS_TESTING_ENVIRONMENT else 180):
        super().__init__()

        self.filename = Path(filename)
        self.save_to_disk_delay = save_to_disk_delay

        self._write_timer = None
        self._lazy_storage = None

        # NOTE: `_last_known_disk_storage_modified_date` is a floating point number giving the
        # number of seconds since the epoch.
        self._last_known_disk_storage_modified_date = -1.0
        self._lazy_disk_cache_size = None

        if not IS_TESTING_ENVIRONMENT:
            atexit.register(self.save)

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

        if self.filename.exists():
            # NOTE: `str(self.filename)` for Python 3.5 support.
            _disk_modified_date = os.path.getmtime(str(self.filename))
            if self._last_known_disk_storage_modified_date < _disk_modified_date:
                with self._write_lock:
                    bytes_ = self.filename.read_bytes()
                    # NOTE: Speed up `pickle.loads` via `gc`, learn more:
                    # https://stackoverflow.com/questions/26860051/how-to-reduce-the-time-taken-to-load-a-pickle-file-in-python
                    gc.disable()
                    disk_storage = pickle.loads(bytes_)
                    gc.enable()

                    logger.info('Loaded `DiskCache` of size %d from `%s`.', len(disk_storage),
                                str(self.filename))

                    self._lazy_storage.update(disk_storage)

                    # NOTE: Only change disk metadata once we're confident we've
                    # ingested the data and won't need to load it again.
                    self._save_disk_metadata(_disk_modified_date, len(disk_storage))

    def _write_timer_save(self, *args, **kwargs):
        """ Alias for `save` that is useful for debugging via tracebacks. """
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
            if not self.filename.exists() or len(self) > self._lazy_disk_cache_size:
                logger.info('Saving `DiskCache` of size %d from `%s`.', len(self),
                            str(self.filename))
                new_disk_storage = pickle.dumps(self._storage)
                self.filename.parent.mkdir(parents=True, exist_ok=True)

                # Atomic write, learn more:
                # http://stupidpythonideas.blogspot.com/2014/07/getting-atomic-writes-right.html
                # https://stackoverflow.com/questions/2333872/atomic-writing-to-file-with-python
                with tempfile.NamedTemporaryFile(dir=TEMP_PATH, delete=False, mode='wb') as file_:
                    file_.write(new_disk_storage)
                os.replace(file_.name, str(self.filename))

                self._save_disk_metadata(os.path.getmtime(str(self.filename)), len(self))

    def set(self, key, value):
        if self.save_to_disk_delay is not None and key not in self:
            if self._write_timer is not None and self._write_timer.is_alive():
                self._write_timer.reset()
            else:
                self._write_timer = src.utils.utils.ResetableTimer(self.save_to_disk_delay,
                                                                   self._write_timer_save)
                self._write_timer.daemon = True
                self._write_timer.start()

        return super().set(key, value)

    def purge(self):
        with self._write_lock:
            if self._write_timer is not None:
                self._write_timer.cancel()

            if self.filename.exists():
                self.filename.unlink()
                self._save_disk_metadata(-1.0, 0)

            if self._lazy_storage is not None:
                super().purge()


def make_arg_key(function, *args, **kwargs):
    """ Get a unique key for `args` and `kwargs` various combinations.

    TODO: Add support variable `*args`.

    Learn more about the implementation:
    https://stackoverflow.com/questions/830937/python-convert-args-to-kwargs

    Args:
        function (callable): Function for which to create the key.
        args (tuple): Arguments passed to `function`.
        kwargs (dict): Keyword arguments passed to `function`.

    Returns:
        frozenset: Set compromising of both `args` and `kwargs`.
    """
    # NOTE: `copy` to avoid changing the provided `kwargs`.
    key = {} if kwargs is None else kwargs.copy()
    if args is not None:
        key.update(zip(function.__code__.co_varnames, args))
    return frozenset(key.items())


_disk_cache_register = []


def get_functions_with_disk_cache():
    """ Get all functions with the `disk_cache` decorator.

    Returns:
        list of callables
    """
    resolved = [i() for i in _disk_cache_register]
    return [i for i in resolved if i is not None]


def disk_cache(function=None,
               directory=DISK_CACHE_PATH,
               save_to_disk_delay=None if IS_TESTING_ENVIRONMENT else 180):
    """ Function decorator that caches function calls and saves them to disk.

    Attrs:
        cache (DiskCache): The function cache.

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

    _use_disk_cache = True
    cache = DiskCache(
        Path(directory) / (inspect.getmodule(function).__name__ + '.' + function.__qualname__),
        save_to_disk_delay)

    @wraps(function)
    def decorator(*args, **kwargs):
        if not _use_disk_cache:
            return function(*args, **kwargs)

        key = make_arg_key(function, *args, **kwargs)
        if key not in cache:
            cache.set(key, function(*args, **kwargs))
        return cache.get(key)

    def use_disk_cache(new_state):
        nonlocal _use_disk_cache
        _use_disk_cache = new_state

    decorator.use_disk_cache = use_disk_cache
    decorator.disk_cache = cache
    _disk_cache_register.append(weakref.ref(decorator))

    return decorator

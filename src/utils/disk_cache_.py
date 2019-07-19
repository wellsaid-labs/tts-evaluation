from functools import partial
from functools import wraps
from pathlib import Path
from threading import RLock
from threading import get_ident

import inspect
import logging
import pickle

from src.utils.utils import HashableDict
from src.utils.utils import ResetableTimer
from src.utils.utils import ROOT_PATH
from src.utils.utils import TTS_DISK_CACHE_NAME

logger = logging.getLogger(__name__)

# NOTE: The file was named `disk_cache_.py` instead of `disk_cache.py` because:
# https://stackoverflow.com/questions/40509588/patch-function-with-same-name-as-module-python-django-mock

LOCK = RLock()


class _Cache(object):
    """ Cache the `args` and `kwargs` with the associated return.

    NOTE: `_Cache` is coupled with `disk_cache` to ensure thread safety.

    Args:
        function (callable)
    """

    def __init__(self, function):
        self._co_varnames = function.__code__.co_varnames
        self._storage = {}

    def _get_key(self, args=(), kwargs={}):
        """
        TODO: Add support variable `*args`.

        Args:
            args (tuple): Arguments passed to `function`.
            kwargs (dict): Keyword arguments password to `function`.

        Returns:
            HashableDict: Dictionary compromising of both `args` and `kwargs`.
        """
        # Learn more: https://stackoverflow.com/questions/830937/python-convert-args-to-kwargs
        key = HashableDict(kwargs.copy())
        key.update(zip(self._co_varnames, args))
        return key

    def set_(self, args=(), kwargs={}, return_=None):
        key = self._get_key(args=args, kwargs=kwargs)
        with LOCK:  # NOTE: Lock `dict` from being mutated
            self._storage[key] = return_

    def get(self, args=(), kwargs={}):
        return self._storage[self._get_key(args=args, kwargs=kwargs)]

    def __iter__(self):
        with LOCK:  # NOTE: Iteration breaks if `self._storage` is mutated during.
            return iter(self._storage)

    def update(self, other_cache):
        if self._co_varnames != other_cache._co_varnames:
            raise ValueError('`other_cache` must be instantiated to the same function.')

        for key in other_cache:
            self.set_(kwargs=key, return_=other_cache.get(kwargs=key))

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


def disk_cache(function=None,
               directory=ROOT_PATH / TTS_DISK_CACHE_NAME / 'disk_cache',
               save_to_disk_delay=60):
    """ Function decorator that caches all function calls and saves them to disk.

    Attrs:
        cache (MutableMapping): The function cache.

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

    name = inspect.getmodule(function).__name__ + '.' + function.__qualname__
    path = Path(directory / name)
    cache = pickle.loads(path.read_bytes()) if path.exists() else _Cache(function)
    timer = None

    def save_to_disk():
        # NOTE Ensure that while writing that the `disk_cache` is not updated; therefore, the
        # `disk_cache` will not lose any data on update.
        print('[%d] save_to_disk waiting for lock' % get_ident())
        with LOCK:
            print('[%d] save_to_disk' % get_ident())
            disk_cache = pickle.loads(path.read_bytes()) if path.exists() else _Cache(function)
            if len(cache) == len(disk_cache):
                return
            cache.update(disk_cache)  # NOTE: `disk_cache` may contain items not in `cache`
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(pickle.dumps(cache))
            print('[%d] save_to_disk done' % get_ident())

    @wraps(function)
    def decorator(*args, **kwargs):
        if (args, kwargs) not in cache:
            cache.set_(args=args, kwargs=kwargs, return_=function(*args, **kwargs))
            print('[%d] not cached %s' % (get_ident(), (args, kwargs)))
        else:
            print('[%d] cached %s' % (get_ident(), (args, kwargs)))

        nonlocal timer
        if timer is not None and timer.is_alive():
            timer.reset()
        else:
            timer = ResetableTimer(save_to_disk_delay, save_to_disk)
            timer.start()

        return cache.get(args=args, kwargs=kwargs)

    # Enable the cache attribute to be accessible.
    decorator.cache = cache

    return decorator

from unittest.mock import patch
from pathlib import Path

import os
import time

import pytest

from src.environment import DISK_CACHE_PATH
from src.utils.disk_cache_ import _Cache
from src.utils.disk_cache_ import disk_cache
from src.utils.disk_cache_ import make_arg_key


def test_make_arg_key():
    function = lambda a, b, c: 'd'
    assert make_arg_key(function, 'a', 'b', c='c') == make_arg_key(function, 'a', 'b', 'c')
    assert make_arg_key(function, 'a', 'b', c='c') == make_arg_key(function, a='a', b='b', c='c')


def test_cache():
    """ Test to ensure basic invariants of `_Cache` are met. """
    cache = _Cache()
    cache.set(1, 'a')

    # `__eq__` basic invariant
    assert cache == cache

    # `__contains__` and `contains` works
    assert 1 in cache

    assert cache.get(1) == 'a'

    assert len(cache) == 1
    assert len(list(iter(cache))) == 1

    # Overwrite fails
    with pytest.raises(ValueError):
        cache.set(1, 'b')

    cache.purge()

    # Purge works
    assert len(cache) == 0
    assert len(list(iter(cache))) == 0
    cache.set(1, 'b')

    assert cache in _Cache.get_instances()


def test_cache__update():
    """ Ensure that `update` works """
    cache = _Cache()
    cache.set(1, 'a')

    other_cache = _Cache()
    other_cache.set(2, 'b')

    cache.update(other_cache)
    assert len(cache) == 2
    assert 1 in cache
    assert 2 in cache


def test_cache__update__overwrite_fails():
    """ Ensure that `update` fails to overwrite. """
    cache = _Cache()
    cache.set(1, 'a')

    other_cache = _Cache()
    other_cache.set(1, 'b')

    with pytest.raises(ValueError):
        cache.update(other_cache)


def test_cache__modify_iterator():
    """ Test that the iterator cannot be modified while iterating. """
    cache = _Cache()
    cache.set(1, 'a')
    cache.set(2, 'b')

    with pytest.raises(RuntimeError):
        for key in cache:
            cache.set(3, 'c')


@patch('src.utils.disk_cache_.os.replace', wraps=os.replace)
def test_disk_cache(mock_replace):
    """ Test to ensure basic invariants of `DiskCache` are met. """

    @disk_cache(directory=DISK_CACHE_PATH, save_to_disk_delay=0.1)
    def helper(arg):
        return arg

    helper('a')
    helper('b')  # Test timer reset

    assert not helper.disk_cache.filename.exists()

    time.sleep(0.2)

    assert helper.disk_cache.filename.exists()
    mock_replace.assert_called_once()
    assert len(helper.disk_cache) == 2
    assert helper.disk_cache.get(make_arg_key(helper.__wrapped__, 'a')) == 'a'
    assert helper.disk_cache.get(make_arg_key(helper.__wrapped__, 'b')) == 'b'

    # Load cache from disk into another decorator instance
    other_helper = disk_cache(helper.__wrapped__, directory=DISK_CACHE_PATH)
    assert other_helper.disk_cache == helper.disk_cache


@patch('src.utils.disk_cache_.os.replace', wraps=os.replace)
def test_disk_cache__redundant_os_operations(mock_replace):
    """ Test to ensure that the `disk_cache` doesn't called `save` or `load` more times than needed.
    """

    @disk_cache(directory=DISK_CACHE_PATH, save_to_disk_delay=0.1)
    def helper(arg):
        return arg

    other_helper = disk_cache(helper.__wrapped__, directory=DISK_CACHE_PATH)

    with patch.object(
            Path, 'read_bytes', wraps=helper.disk_cache.filename.read_bytes) as mock_read_bytes:
        for i in range(100):  # Ensure that `save_to_disk_delay` works.
            helper('a' * i)

        helper.disk_cache.save()  # Cancel the write timer and save.
        time.sleep(0.2)  # Enough time for `save_to_disk_delay` to expire.
        helper.disk_cache.save()  # Try multiple saves in a row.
        assert mock_replace.call_count == 1

        helper.disk_cache.load()
        assert mock_read_bytes.call_count == 0

        helper('a')  # Test if redundant call triggers a save, it shouldn't.
        time.sleep(0.2)
        assert mock_replace.call_count == 1

        helper('b')  # New data should trigger a save.
        time.sleep(0.2)
        assert mock_replace.call_count == 2
        assert mock_read_bytes.call_count == 0

    with patch.object(
            Path, 'read_bytes',
            wraps=other_helper.disk_cache.filename.read_bytes) as mock_read_bytes:
        assert other_helper.disk_cache == helper.disk_cache
        assert mock_read_bytes.call_count == 1

    with patch.object(
            Path, 'read_bytes', wraps=helper.disk_cache.filename.read_bytes) as mock_read_bytes:
        other_helper('c')
        other_helper.disk_cache.save()
        assert mock_replace.call_count == 3
        helper.disk_cache.save()  # Loads new data from `other_helper`.
        assert mock_read_bytes.call_count == 1

        helper.disk_cache.purge()

    with patch.object(
            Path, 'read_bytes',
            wraps=other_helper.disk_cache.filename.read_bytes) as mock_read_bytes:
        other_helper.disk_cache.save()
        assert mock_replace.call_count == 4
        assert mock_read_bytes.call_count == 0


@patch('src.utils.disk_cache_.os.replace', wraps=os.replace)
def test_disk_cache__delete_cache(mock_replace):
    """ Test to ensure that cache is able to handle the disk cache getting deleted.
    """

    @disk_cache(directory=DISK_CACHE_PATH, save_to_disk_delay=0.1)
    def helper(arg):
        return arg

    helper('a')
    helper.disk_cache.save()
    assert mock_replace.call_count == 1
    helper.disk_cache.save()
    assert mock_replace.call_count == 1
    helper.disk_cache.filename.unlink()
    helper.disk_cache.save()
    assert mock_replace.call_count == 2

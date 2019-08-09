from mock import patch
from pathlib import Path

import os
import time

import pytest

from src.environment import TEMP_PATH
from src.utils.disk_cache_ import _Cache
from src.utils.disk_cache_ import disk_cache


def test_cache():
    """ Test to ensure basic invariants of `_Cache` are met. """
    cache = _Cache(lambda a, b, c: 'd')
    cache.set(args=('a', 'b'), kwargs={'c': 'c'}, return_='d')

    # `__eq__` basic invariant
    assert cache == cache

    # `__contains__` and `contains` works
    assert ('a', 'b', 'c') in cache
    assert {'a': 'a', 'b': 'b', 'c': 'c'} in cache
    assert cache.contains(args=('a', 'b'), kwargs={'c': 'c'})

    assert cache.get(kwargs={'a': 'a', 'b': 'b', 'c': 'c'}) == 'd'

    assert len(cache) == 1
    assert len(list(iter(cache))) == 1

    # Overwrite fails
    with pytest.raises(ValueError):
        cache.set(args=('a', 'b'), kwargs={'c': 'c'}, return_='e')

    cache.purge()

    # Purge works
    assert len(cache) == 0
    assert len(list(iter(cache))) == 0
    cache.set(args=('a', 'b'), kwargs={'c': 'c'}, return_='e')

    assert cache in _Cache.get_instances()


def test_cache__update():
    """ Ensure that `update` works """
    function = lambda a, b, c: 'd'

    cache = _Cache(function)
    cache.set(args=('a', 'b'), kwargs={'c': 'c'}, return_='d')

    other_cache = _Cache(function)
    other_cache.set(args=('a', 'b'), kwargs={'c': None}, return_='d')

    cache.update(other_cache)
    assert len(cache) == 2
    assert ('a', 'b', 'c') in cache
    assert ('a', 'b', None) in cache


def test_cache__not_equal():
    """ Ensure that not `equal` works. """
    function = lambda a, b, c: 'd'
    other_function = lambda a, c: 'd'

    assert _Cache(function) != _Cache(other_function)


def test_cache__update__invalid():
    """ Ensure that `update` works. """
    function = lambda a, b, c: 'd'
    other_function = lambda a, c: 'd'

    cache = _Cache(function)
    cache.set(args=('a', 'b'), kwargs={'c': 'c'}, return_='d')

    other_cache = _Cache(other_function)
    other_cache.set(args=('a', 'b'), return_='d')

    with pytest.raises(ValueError):
        cache.update(other_cache)


def test_cache__update__overwrite_fails():
    """ Ensure that `update` works """
    function = lambda a, b, c: 'd'

    cache = _Cache(function)
    cache.set(args=('a', 'b'), kwargs={'c': 'c'}, return_='d')

    other_cache = _Cache(function)
    other_cache.set(args=('a', 'b'), kwargs={'c': 'c'}, return_='e')

    with pytest.raises(ValueError):
        cache.update(other_cache)


def test_cache__modify_iterator():
    """ Test that the iterator cannot be modified while iterating. """
    function = lambda a, b, c: 'd'
    cache = _Cache(function)
    cache.set(args=('a', 'b'), kwargs={'c': 'c'}, return_='d')
    cache.set(args=('a', 'b'), kwargs={'d': 'd'}, return_='d')

    with pytest.raises(RuntimeError):
        for key in cache:
            cache.set(args=('a', 'b'), kwargs={'e': 'e'}, return_='d')


@patch('src.utils.disk_cache_.os.replace', wraps=os.replace)
def test_disk_cache(mock_replace):
    """ Test to ensure basic invariants of `_DiskCache` are met. """
    file_name = 'tests.utils.test_disk_cache_.test_disk_cache.<locals>.helper'
    file_path = TEMP_PATH / '.disk_cache' / file_name

    @disk_cache(directory=file_path.parent, save_to_disk_delay=0.1)
    def helper(arg):
        return arg

    helper('a')
    helper('b')  # Test timer reset

    assert not file_path.exists()

    time.sleep(0.4)

    assert file_path.exists()
    mock_replace.assert_called_once()
    assert len(helper.cache) == 2
    assert helper.cache.get(kwargs={'arg': 'a'}) == 'a'
    assert helper.cache.get(kwargs={'arg': 'b'}) == 'b'

    # Load cache from disk into another decorator instance
    other_helper = disk_cache(helper.__wrapped__, directory=file_path.parent)
    assert other_helper.cache == helper.cache


@patch('src.utils.disk_cache_.os.replace', wraps=os.replace)
def test_disk_cache__redundant_os_operations(mock_replace):
    """ Test to ensure that the `disk_cache` doesn't called `save` or `load` more times than needed.
    """
    file_name = 'tests.utils.test_disk_cache_.test_disk_cache.<locals>.helper'
    file_path = TEMP_PATH / '.disk_cache' / file_name

    @disk_cache(directory=file_path.parent, save_to_disk_delay=0.1)
    def helper(arg):
        return arg

    other_helper = disk_cache(helper.__wrapped__, directory=file_path.parent)

    with patch.object(
            Path, 'read_bytes', wraps=helper.cache._file_path.read_bytes) as mock_read_bytes:
        for i in range(10000):  # Ensure that `save_to_disk_delay` works.
            helper('a' * i)

        helper.cache.save()  # Cancel the write timer and save.
        time.sleep(0.4)  # Enough time for `save_to_disk_delay` to expire.
        helper.cache.save()  # Try multiple saves in a row.
        assert mock_replace.call_count == 1

        helper.cache.load()
        assert mock_read_bytes.call_count == 0

        helper('a')  # Test if redundant call triggers a save, it shouldn't.
        time.sleep(0.4)
        assert mock_replace.call_count == 1

        helper('b')  # New data should trigger a save.
        time.sleep(0.4)
        assert mock_replace.call_count == 2
        assert mock_read_bytes.call_count == 0

    with patch.object(
            Path, 'read_bytes', wraps=other_helper.cache._file_path.read_bytes) as mock_read_bytes:
        assert other_helper.cache == helper.cache
        assert mock_read_bytes.call_count == 1

    with patch.object(
            Path, 'read_bytes', wraps=helper.cache._file_path.read_bytes) as mock_read_bytes:
        other_helper('c')
        other_helper.cache.save()
        assert mock_replace.call_count == 3
        helper.cache.save()  # Loads new data from `other_helper`.
        assert mock_read_bytes.call_count == 1

        helper.cache.purge()

    with patch.object(
            Path, 'read_bytes', wraps=other_helper.cache._file_path.read_bytes) as mock_read_bytes:
        other_helper.cache.save()
        assert mock_replace.call_count == 4
        assert mock_read_bytes.call_count == 0


@patch('src.utils.disk_cache_.os.replace', wraps=os.replace)
def test_disk_cache__delete_cache(mock_replace):
    """ Test to ensure that cache is able to handle the disk cache getting deleted.
    """
    file_name = 'tests.utils.test_disk_cache_.test_disk_cache.<locals>.helper'
    file_path = TEMP_PATH / '.disk_cache' / file_name

    @disk_cache(directory=file_path.parent, save_to_disk_delay=0.1)
    def helper(arg):
        return arg

    helper('a')
    helper.cache.save()
    assert mock_replace.call_count == 1
    helper.cache.save()
    assert mock_replace.call_count == 1
    helper.cache._file_path.unlink()
    helper.cache.save()
    assert mock_replace.call_count == 2

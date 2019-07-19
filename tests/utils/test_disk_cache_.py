import time

from src.environment import ROOT_PATH
from src.utils.disk_cache_ import _Cache
from src.utils.disk_cache_ import disk_cache


def test_cache():
    cache = _Cache(lambda a, b, c: 'd')
    cache.set_(args=('a', 'b'), kwargs={'c': 'c'}, return_='d')
    assert (('a', 'b', 'c'),) in cache
    assert (('a', 'b'), {'c': 'c'}) in cache
    assert cache.get(kwargs={'a': 'a', 'b': 'b', 'c': 'c'}) == 'd'
    assert len(cache) == 1
    assert len(list(iter(cache))) == 1


def test_cache_update():
    function = lambda a, b, c: 'd'

    cache = _Cache(function)
    cache.set_(args=('a', 'b'), kwargs={'c': 'c'}, return_='d')

    other_cache = _Cache(function)
    other_cache.set_(args=('a', 'b'), kwargs={'c': None}, return_='d')

    cache.update(other_cache)
    assert len(cache) == 2
    assert (('a', 'b', 'c'),) in cache
    assert (('a', 'b', None),) in cache


def test_disk_cache():
    filename = (
        ROOT_PATH / 'tests' / '_test_data' / '.disk_cache' /
        'tests.utils.test_disk_cache_.test_disk_cache.<locals>.helper')

    @disk_cache(directory=filename.parent, save_to_disk_delay=.25)
    def helper(arg):
        return arg

    helper('a')
    helper('b')  # Trigger timer reset

    assert not filename.exists()

    time.sleep(0.3)

    assert filename.exists()
    # TODO: Ensure `write_bytes` was only called once via an `assert`
    assert len(helper.cache) == 2
    assert helper.cache.get(kwargs={'arg': 'a'}) == 'a'
    assert helper.cache.get(kwargs={'arg': 'b'}) == 'b'

    # Load cache from disk into another decorator instance
    other_helper = disk_cache(helper, directory=filename.parent)
    assert other_helper.cache == helper.cache

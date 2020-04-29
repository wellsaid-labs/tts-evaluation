# Fix this weird error:
# https://stackoverflow.com/questions/37604289/tkinter-tclerror-no-display-name-and-no-display-environment-variable
import matplotlib
matplotlib.use('Agg')

from hparams import clear_config

# Fix this weird error: https://github.com/pytorch/pytorch/issues/2083
import torch  # noqa: F401

import pytest

from hparams import set_lazy_resolution

from src.environment import set_basic_logging_config
from src.environment import TEST_DATA_PATH
from src.hparams import set_hparams
from src.utils.disk_cache_ import DiskCache
from tests._utils import create_disk_garbage_collection_fixture

set_basic_logging_config()


@pytest.fixture(autouse=True)
def run_before_test():
    # Invalidate cache before each test.
    clear_config()
    for cache in DiskCache.get_instances():
        cache.purge()

    set_lazy_resolution(True)  # This helps performance for individual tests
    set_hparams()

    with torch.autograd.detect_anomaly():
        yield

    # NOTE: We need to invalidate caching after the test because of delayed writes.
    for cache in DiskCache.get_instances():
        cache.purge()


gc_fixture_test_data = create_disk_garbage_collection_fixture(TEST_DATA_PATH, autouse=True)

# Fix this weird error:
# https://stackoverflow.com/questions/37604289/tkinter-tclerror-no-display-name-and-no-display-environment-variable
import matplotlib
matplotlib.use('Agg')

# Fix this weird error: https://github.com/pytorch/pytorch/issues/2083
import torch  # noqa: F401

import pytest

from src.environment import ROOT_PATH
from src.environment import set_basic_logging_config
from src.environment import TTS_DISK_CACHE_NAME
from src.hparams import clear_config
from src.hparams import set_hparams
from src.utils.disk_cache_ import _DiskCache
from tests._utils import create_disk_garbage_collection_fixture

set_basic_logging_config()


@pytest.fixture(autouse=True)
def run_before_test():
    # Invalidate cache before each test.
    clear_config()
    for cache in _DiskCache.get_instances():
        cache.clear()

    set_hparams()

    yield


(ROOT_PATH / TTS_DISK_CACHE_NAME).mkdir(exist_ok=True)

gc_fixture_tts_cache = create_disk_garbage_collection_fixture(
    ROOT_PATH / TTS_DISK_CACHE_NAME, autouse=True)

gc_fixture_test_data = create_disk_garbage_collection_fixture(
    ROOT_PATH / 'tests' / '_test_data', autouse=True)

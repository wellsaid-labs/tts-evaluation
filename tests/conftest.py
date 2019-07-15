# Fix this weird error:
# https://stackoverflow.com/questions/37604289/tkinter-tclerror-no-display-name-and-no-display-environment-variable
import matplotlib
matplotlib.use('Agg')

# Fix this weird error: https://github.com/pytorch/pytorch/issues/2083
import torch  # noqa: F401

import logging

import pytest

from src.hparams import clear_config
from src.hparams import set_hparams
from src.utils import ROOT_PATH
from src.utils import TTS_DISK_CACHE_NAME
from tests._utils import create_disk_garbage_collection_fixture

logging.getLogger().setLevel(logging.INFO)


@pytest.fixture(autouse=True)
def run_before_test():
    clear_config()
    set_hparams()
    yield
    clear_config()


gc_fixture_test_data = create_disk_garbage_collection_fixture(
    ROOT_PATH / 'tests' / '_test_data', autouse=True)

if (ROOT_PATH / TTS_DISK_CACHE_NAME).exists():
    gc_fixture_tts_cache = create_disk_garbage_collection_fixture(
        ROOT_PATH / TTS_DISK_CACHE_NAME, autouse=True)

if (ROOT_PATH / TTS_DISK_CACHE_NAME / 'disk_cache').exists():
    gc_fixture_disk_cache = create_disk_garbage_collection_fixture(
        ROOT_PATH / TTS_DISK_CACHE_NAME / 'disk_cache', autouse=True)

gc_fixture_experiments = create_disk_garbage_collection_fixture(
    ROOT_PATH / 'experiments', autouse=True)

gc_fixture_root = create_disk_garbage_collection_fixture(ROOT_PATH, autouse=True)

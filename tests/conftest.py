import warnings

# Fix this weird error:
# https://stackoverflow.com/questions/37604289/tkinter-tclerror-no-display-name-and-no-display-environment-variable
import matplotlib
matplotlib.use('Agg')

# Fix this weird error: https://github.com/pytorch/pytorch/issues/2083
import torch  # noqa: F401

import pytest

from hparams import set_lazy_resolution

from lib.environment import set_basic_logging_config
from lib.environment import TEST_DATA_PATH
from tests._utils import create_disk_garbage_collection_fixture

set_basic_logging_config()


@pytest.fixture(autouse=True)
def run_before_test():
    set_lazy_resolution(True)  # NOTE: This improves performance for `hparams`.

    with warnings.catch_warnings():
        warnings.filterwarnings(
            'ignore', module=r'.*torch.*', message=r'.*Anomaly Detection has been enabled.*')
        with torch.autograd.detect_anomaly():
            yield


gc_fixture_test_data = create_disk_garbage_collection_fixture(TEST_DATA_PATH, autouse=True)

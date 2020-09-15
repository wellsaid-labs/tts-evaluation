import warnings

# Fix this weird error:
# https://stackoverflow.com/questions/37604289/tkinter-tclerror-no-display-name-and-no-display-environment-variable
import matplotlib
matplotlib.use('Agg')

# Fix this weird error: https://github.com/pytorch/pytorch/issues/2083
import torch  # noqa: F401

from hparams import set_lazy_resolution

import pytest

import lib

lib.environment.set_basic_logging_config()


@pytest.fixture(autouse=True)
def run_before_test():
    set_lazy_resolution(True)  # NOTE: This improves performance for `hparams`.

    with warnings.catch_warnings():
        warnings.filterwarnings(
            'ignore', module=r'.*torch.*', message=r'.*Anomaly Detection has been enabled.*')
        with torch.autograd.detect_anomaly():
            yield

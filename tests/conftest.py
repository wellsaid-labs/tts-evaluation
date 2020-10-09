# Fix this weird error:
# https://stackoverflow.com/questions/37604289/tkinter-tclerror-no-display-name-and-no-display-environment-variable
import matplotlib  # isort: skip

matplotlib.use("Agg")

# Fix this weird error: https://github.com/pytorch/pytorch/issues/2083
import torch  # noqa: F401, E402  # isort: skip

import warnings  # noqa: E402

import pytest  # noqa: E402
from hparams import set_lazy_resolution  # noqa: E402

import lib  # noqa: E402

lib.environment.set_basic_logging_config()


@pytest.fixture(autouse=True)
def run_before_test():
    set_lazy_resolution(True)  # NOTE: This improves performance for `hparams`.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            module=r".*torch.*",
            message=r".*Anomaly Detection has been enabled.*",
        )
        with torch.autograd.detect_anomaly():
            yield

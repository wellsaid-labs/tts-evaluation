# Fix this weird error:
# https://stackoverflow.com/questions/37604289/tkinter-tclerror-no-display-name-and-no-display-environment-variable
import matplotlib  # isort: skip

matplotlib.use("Agg")

# Fix this weird error: https://github.com/pytorch/pytorch/issues/2083
import torch  # noqa: F401, E402  # isort: skip
import warnings  # noqa: E402

import config as cf  # noqa: E402
import pytest  # noqa: E402
import torch.autograd  # noqa: E402
import torch.distributed  # noqa: E402

import lib  # noqa: E402

lib.environment.set_basic_logging_config()

# TODO: Support doctests, so our documentation remains accurate.


@pytest.fixture(autouse=True, scope="session")
def run_around_session():
    cf.enable_fast_trace()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r".*Anomaly Detection has been enabled.*")
        with torch.autograd.anomaly_mode.detect_anomaly():
            yield
    try:
        torch.distributed.destroy_process_group()
    except (RuntimeError, AssertionError):
        pass


@pytest.fixture(autouse=True)
def run_around_test():
    yield
    try:
        torch.distributed.destroy_process_group()
    except (RuntimeError, AssertionError):
        pass

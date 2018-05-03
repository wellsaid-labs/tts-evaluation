# Fix this weird error: https://github.com/pytorch/pytorch/issues/2083
import torch  # noqa: F401

import pytest

from src.utils import config_logging
from src.hparams import set_hparams

set_hparams()
config_logging()


def pytest_configure(config):
    """Inject documentation."""
    config.addinivalue_line("markers", "capture_disabled: ")


@pytest.hookimpl(hookwrapper=True)
def pytest_pyfunc_call(pyfuncitem):
    if 'capture_disabled' in pyfuncitem.keywords:
        capmanager = pyfuncitem._request.config.pluginmanager.getplugin('capturemanager')
        capmanager.suspend_capture_item(pyfuncitem._request.node, "call", in_=True)

        print('')
        print('')
        print('======= Capsys Disabled: {} ======= '.format(pyfuncitem.obj.__name__))

    try:
        yield
    finally:
        if 'capture_disabled' in pyfuncitem.keywords:
            print('========= Capsys Enabled ========= ')
            print('')


# REFERENCE: https://docs.pytest.org/en/latest/example/simple.html


def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true", default=False, help="run slow tests")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        # --runslow given in cli: do not skip slow tests
        return
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)

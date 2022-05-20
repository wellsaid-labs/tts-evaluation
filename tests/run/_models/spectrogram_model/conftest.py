import config as cf
import pytest

import run


@pytest.fixture(autouse=True, scope="module")
def run_around_tests():
    """Set a basic configuration."""
    run._config.configure()
    yield
    cf.purge()

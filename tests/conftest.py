# Fix this weird error:
# https://stackoverflow.com/questions/37604289/tkinter-tclerror-no-display-name-and-no-display-environment-variable
import matplotlib
matplotlib.use('Agg')

# NOTE: Comet needs to be imported before torch
import comet_ml  # noqa: F401

# Fix this weird error: https://github.com/pytorch/pytorch/issues/2083
import torch  # noqa: F401

import logging

import pytest

from src.hparams import set_hparams
from src.hparams import clear_config

logging.getLogger().setLevel(logging.INFO)


@pytest.fixture(autouse=True)
def run_before_test():
    clear_config()
    set_hparams()
    yield
    clear_config()

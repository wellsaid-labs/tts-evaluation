from functools import lru_cache

import ctypes
import logging
import logging.config
import os
import time
import sys

import random
import torch
import numpy as np

logger = logging.getLogger(__name__)


def config_logging():
    """ Configure the root logger with basic settings.
    """
    logging.basicConfig(
        format='[%(asctime)s][%(processName)s][%(name)s][%(levelname)s] %(message)s',
        level=logging.INFO,
        stream=sys.stdout)

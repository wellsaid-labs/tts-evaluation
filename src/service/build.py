""" Builds signal model inferrer and cache. """
import json
import pathlib

from src.environment import ROOT_PATH
from src.utils import Checkpoint

model_config = json.loads((ROOT_PATH / 'src' / 'service' / 'models.config.json').read_text())
SIGNAL_MODEL_CHECKPOINT_PATH = pathlib.Path(model_config['signal_model'])
assert SIGNAL_MODEL_CHECKPOINT_PATH.is_file(), 'Signal model checkpoint cannot be found.'

Checkpoint.from_path(SIGNAL_MODEL_CHECKPOINT_PATH).model.to_inferrer()  # Cache inferrer build

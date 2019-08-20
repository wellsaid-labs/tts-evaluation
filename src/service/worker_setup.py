""" Runs various preprocessing code to build the `docker/worker/Dockerfile` file.
"""
import json
import pathlib

from src.environment import ROOT_PATH
from src.environment import set_basic_logging_config
from src.utils import Checkpoint

set_basic_logging_config()

model_config = json.loads((ROOT_PATH / 'src' / 'service' / 'models.config.json').read_text())

signal_model_checkpoint_path = pathlib.Path(model_config['signal_model'])
spectrogram_model_checkpoint_path = pathlib.Path(model_config['spectrogram_model'])

assert signal_model_checkpoint_path.is_file(), 'Signal model checkpoint cannot be found.'
assert spectrogram_model_checkpoint_path.is_file(), 'Spectrogram model checkpoint cannot be found.'

signal_model_checkpoint = Checkpoint.from_path(signal_model_checkpoint_path)
spectrogram_model_checkpoint = Checkpoint.from_path(spectrogram_model_checkpoint_path)

# Cache ths signal model inferrer
signal_model_checkpoint.model.to_inferrer()

# TODO: The below checkpoint attributes should be statically defined somewhere so that there is some
# guarantee that these attributes exist.

# Reduce checkpoint size
signal_model_checkpoint.optimizer = None
signal_model_checkpoint.anomaly_detector = None
spectrogram_model_checkpoint.optimizer = None

# Remove unnecessary information
signal_model_checkpoint.comet_ml_project_name = None
signal_model_checkpoint.comet_ml_experiment_key = None
spectrogram_model_checkpoint.comet_ml_project_name = None
spectrogram_model_checkpoint.comet_ml_experiment_key = None

spectrogram_model_checkpoint.save()
signal_model_checkpoint.save()

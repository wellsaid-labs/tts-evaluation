""" Configuration outside of Kubernetes or Docker.

This allows both `worker_setup.py` and `worker.py` to be run outside of Docker and Kubernetes.

TODO: These models should be uploaded online for so that everyone has access to them, and we do not
lose track of them.
"""
from src.environment import IS_TESTING_ENVIRONMENT
from src.environment import ROOT_PATH

SPECTROGRAM_MODEL_CHECKPOINT_PATH = ROOT_PATH / 'experiments/spectrogram_model/step_183968.pt'
SIGNAL_MODEL_CHECKPOINT_PATH = ROOT_PATH / 'experiments/signal_model/step_165137.pt'

if not IS_TESTING_ENVIRONMENT:
    assert SIGNAL_MODEL_CHECKPOINT_PATH.is_file(), 'Signal model checkpoint cannot be found.'
    assert SPECTROGRAM_MODEL_CHECKPOINT_PATH.is_file(
    ), 'Spectrogram model checkpoint cannot be found.'

SPEAKER_ID_TO_SPEAKER_ID = {
    0: 4,  # Judy Bieber
    1: 2,  # Mary Ann
    2: 3,  # Linda Johnson
    3: 5,  # Hilary Noriega
    4: 0,  # Beth Cameron
    5: 1,  # Beth Cameron (Custom)
    6: 3,  # Linda Johnson
    7: 6,  # Sam Scholl
    8: 7,  # Adrienne Walker-Heller
    9: 8,  # Frank Bonacquisti
    10: 9,  # Susan Murphy
    11: 10,  # Heather Doe
}

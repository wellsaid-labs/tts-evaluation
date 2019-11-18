""" Configuration outside of Kubernetes or Docker.

This allows both `worker_setup.py` and `worker.py` to be run outside of Docker and Kubernetes.

TODO: These models should be uploaded online for so that everyone has access to them, and we do not
lose track of them.
"""
from src.environment import IS_TESTING_ENVIRONMENT
from src.environment import SIGNAL_MODEL_EXPERIMENTS_PATH
from src.environment import SPECTROGRAM_MODEL_EXPERIMENTS_PATH

SPECTROGRAM_MODEL_CHECKPOINT_PATH = SPECTROGRAM_MODEL_EXPERIMENTS_PATH / 'MY_LOCAL_SPECTROGRAM.pt'
SIGNAL_MODEL_CHECKPOINT_PATH = SIGNAL_MODEL_EXPERIMENTS_PATH / 'MY_LOCAL_SIGNAL.pt'

if not IS_TESTING_ENVIRONMENT:
    assert SIGNAL_MODEL_CHECKPOINT_PATH.is_file(), 'Signal model checkpoint cannot be found.'
    assert SPECTROGRAM_MODEL_CHECKPOINT_PATH.is_file(
    ), 'Spectrogram model checkpoint cannot be found.'

# NOTE: These value (not the keys) need to be updated based on the spectrogram model encoder.
# The keys need to stay the same for backwards compatibility.
SPEAKER_ID_TO_SPEAKER_ID = {
    0: 6,  # Judy Bieber
    1: 13,  # Mary Ann
    2: 14,  # Linda Johnson
    3: 10,  # Hilary Noriega
    4: 9,  # Beth Cameron
    5: 3,  # Beth Cameron (Custom)
    6: 14,  # Linda Johnson
    7: 1,  # Sam Scholl
    8: 18,  # Adrienne Walker-Heller
    9: 8,  # Frank Bonacquisti
    10: 5,  # Susan Murphy
    11: 11,  # Heather Doe
    12: 12,  # Alicia Harris
    13: 2,  # George Drake
    14: 4,  # Megan Sinclair
    15: 15,  # Elise Randall
    16: 7,  # Hanuman Welch
    17: 16,  # Jack Rutkowski
    18: 17,  # Mark Atherlay
    19: 0,  # Steven Wahlberg
}

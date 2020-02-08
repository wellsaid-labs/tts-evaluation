""" Configuration outside of Kubernetes or Docker.

This allows both `worker_setup.py` and `worker.py` to be run outside of Docker and Kubernetes.

TODO: These models should be uploaded online for so that everyone has access to them, and we do not
lose track of them.
"""
from src import datasets
from src.environment import IS_TESTING_ENVIRONMENT
from src.environment import SIGNAL_MODEL_EXPERIMENTS_PATH
from src.environment import SPECTROGRAM_MODEL_EXPERIMENTS_PATH

SPECTROGRAM_MODEL_CHECKPOINT_PATH = SPECTROGRAM_MODEL_EXPERIMENTS_PATH / 'rhyan_viacom_poc_ALL_TRANSFER_LEARNING_36k_experiments_spectrogram_model_DATE-2020y01m27d-06h16m55s_PID-32774_RUN_DATE-2020y01m27d-06h16m55s_checkpoints_step_36736.pt'
SIGNAL_MODEL_CHECKPOINT_PATH = SIGNAL_MODEL_EXPERIMENTS_PATH / 'rhyan_viacom_poc_ALL_TRANSFER_LEARNING_36k_experiments_signal_model_standard_learning_DATE-2020y02m05d-02h37m24s_PID-28231_RUN_DATE-2020y02m05d-02h37m24s_checkpoints_step_169079.pt'

if not IS_TESTING_ENVIRONMENT:
    assert SIGNAL_MODEL_CHECKPOINT_PATH.is_file(), 'Signal model checkpoint cannot be found.'
    assert SPECTROGRAM_MODEL_CHECKPOINT_PATH.is_file(
    ), 'Spectrogram model checkpoint cannot be found.'

# NOTE: These value (not the keys) need to be updated based on the spectrogram model encoder.
# The keys need to stay the same for backwards compatibility.
SPEAKER_ID_TO_SPEAKER = {
    0: datasets.JUDY_BIEBER,
    1: datasets.MARY_ANN,
    2: datasets.LINDA_JOHNSON,
    3: datasets.HILARY_NORIEGA,
    4: datasets.BETH_CAMERON,
    5: datasets.BETH_CAMERON_CUSTOM,
    6: datasets.LINDA_JOHNSON,
    7: datasets.SAM_SCHOLL,
    8: datasets.ADRIENNE_WALKER_HELLER,
    9: datasets.FRANK_BONACQUISTI,
    10: datasets.SUSAN_MURPHY,
    11: datasets.HEATHER_DOE,
    12: datasets.ALICIA_HARRIS,
    13: datasets.GEORGE_DRAKE,
    14: datasets.MEGAN_SINCLAIR,
    15: datasets.ELISE_RANDALL,
    16: datasets.HANUMAN_WELCH,
    17: datasets.JACK_RUTKOWSKI,
    18: datasets.MARK_ATHERLAY,
    19: datasets.STEVEN_WAHLBERG,

    # NOTE: Custom voice IDs are random numbers larger than 10,000 and less than 20,000.
    11541: datasets.LINCOLN_CUSTOM
}

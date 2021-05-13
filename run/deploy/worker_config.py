""" Configuration outside of Kubernetes or Docker.

This allows both `worker_setup.py` and `worker.py` to be run outside of Docker and Kubernetes.

TODO: These models should be uploaded online for so that everyone has access to them, and we do not
lose track of them.
"""
from run._config import SIGNAL_MODEL_EXPERIMENTS_PATH, SPECTROGRAM_MODEL_EXPERIMENTS_PATH
from run.data import _loader

SPECTROGRAM_MODEL_CHECKPOINT_PATH = SPECTROGRAM_MODEL_EXPERIMENTS_PATH / "MY_LOCAL_SPECTROGRAM.pt"
SIGNAL_MODEL_CHECKPOINT_PATH = SIGNAL_MODEL_EXPERIMENTS_PATH / "MY_LOCAL_SIGNAL.pt"


# NOTE: These value (not the keys) need to be updated based on the spectrogram model encoder.
# The keys need to stay the same for backwards compatibility.
SPEAKER_ID_TO_SPEAKER = {
    0: _loader.JUDY_BIEBER,
    1: _loader.MARY_ANN,
    2: _loader.LINDA_JOHNSON,
    3: _loader.HILARY_NORIEGA,
    4: _loader.BETH_CAMERON,
    5: _loader.BETH_CAMERON__CUSTOM,
    6: _loader.LINDA_JOHNSON,
    7: _loader.SAM_SCHOLL,
    8: _loader.ADRIENNE_WALKER_HELLER,
    9: _loader.FRANK_BONACQUISTI,
    10: _loader.SUSAN_MURPHY,
    11: _loader.HEATHER_DOE,
    12: _loader.ALICIA_HARRIS,
    13: _loader.GEORGE_DRAKE_JR,
    14: _loader.MEGAN_SINCLAIR,
    15: _loader.ELISE_RANDALL,
    16: _loader.HANUMAN_WELCH,
    17: _loader.JACK_RUTKOWSKI,
    18: _loader.MARK_ATHERLAY,
    19: _loader.STEVEN_WAHLBERG,
    20: _loader.ADRIENNE_WALKER_HELLER__PROMO,
    21: _loader.DAMON_PAPADOPOULOS__PROMO,
    22: _loader.DANA_HURLEY__PROMO,
    23: _loader.ED_LACOMB__PROMO,
    24: _loader.LINSAY_ROUSSEAU__PROMO,
    25: _loader.MARI_MONGE__PROMO,
    26: _loader.SAM_SCHOLL__PROMO,
    27: _loader.JOHN_HUNERLACH__NARRATION,
    28: _loader.JOHN_HUNERLACH__RADIO,
    29: _loader.OTIS_JIRY__STORY,
    # NOTE: Custom voice IDs are random numbers larger than 10,000...
    11541: _loader.LINCOLN__CUSTOM,
    13268907: _loader.JOSIE__CUSTOM,
    95313811: _loader.JOSIE__CUSTOM__MANUAL_POST,
    78252076: ...,  # TODO: Add Veritone Custom Voice
    70695443: ...,  # TODO: Add Super Hi-Fi Custom Voice
}

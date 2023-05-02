# NOTE: Other `run` modules import `lang`; therefore, they need to be imported first.
from run._config import lang  # isort: skip

from run._config import audio, data, environment, labels
from run._config.all import configure
from run._config.audio import FRAME_HOP, NUM_FRAME_CHANNELS
from run._config.data import DATASETS, DEFAULT_SCRIPT, DEV_SPEAKERS
from run._config.environment import (
    CHECKPOINTS_PATH,
    DATA_PATH,
    DATASET_CACHE_PATH,
    RANDOM_SEED,
    SAMPLES_PATH,
    SIG_MODEL_EXP_PATH,
    SPEC_MODEL_EXP_PATH,
    TEMP_PATH,
    TTS_PACKAGE_PATH,
)
from run._config.labels import (
    Cadence,
    DatasetType,
    Device,
    GetLabel,
    Label,
    get_config_label,
    get_dataset_label,
    get_environment_label,
    get_model_label,
    get_signal_model_label,
    get_timer_label,
)
from run._config.lang import (
    RESPELLING_DELIM,
    STT_CONFIGS,
    LanguageCode,
    get_spoken_chars,
    is_normalized_vo_script,
    is_sound_alike,
    is_voiced,
    load_spacy_nlp,
    normalize_and_verbalize_text,
    normalize_vo_script,
    replace_punc,
)
from run._config.train import (
    config_fine_tune_training,
    config_sig_model_training_from_datasets,
    config_spec_model_training_from_datasets,
)

# TODO: Reduce the usage of globals, and use configuration if possible.

__all__ = [
    "audio",
    "data",
    "environment",
    "labels",
    "lang",
    "configure",
    "FRAME_HOP",
    "NUM_FRAME_CHANNELS",
    "DATASETS",
    "DEFAULT_SCRIPT",
    "DEV_SPEAKERS",
    "CHECKPOINTS_PATH",
    "DATA_PATH",
    "DATASET_CACHE_PATH",
    "RANDOM_SEED",
    "SAMPLES_PATH",
    "SIG_MODEL_EXP_PATH",
    "SPEC_MODEL_EXP_PATH",
    "TEMP_PATH",
    "TTS_PACKAGE_PATH",
    "Cadence",
    "DatasetType",
    "Device",
    "GetLabel",
    "Label",
    "get_config_label",
    "get_dataset_label",
    "get_environment_label",
    "get_model_label",
    "get_signal_model_label",
    "get_timer_label",
    "RESPELLING_DELIM",
    "STT_CONFIGS",
    "LanguageCode",
    "get_spoken_chars",
    "is_normalized_vo_script",
    "is_sound_alike",
    "is_voiced",
    "normalize_vo_script",
    "replace_punc",
    "normalize_and_verbalize_text",
    "load_spacy_nlp",
    "config_fine_tune_training",
    "config_sig_model_training_from_datasets",
    "config_spec_model_training_from_datasets",
]

import typing

from run.data._loader import data_structures, m_ailabs, utils
from run.data._loader.data_structures import (
    Alignment,
    NonalignmentSpans,
    Passage,
    Span,
    Speaker,
    alignment_dtype,
    has_a_mistranscription,
    voiced_nonalignment_spans,
)
from run.data._loader.lj_speech import LINDA_JOHNSON, lj_speech_dataset
from run.data._loader.m_ailabs import (
    ELIZABETH_KLETT,
    ELLIOT_MILLER,
    JUDY_BIEBER,
    MARY_ANN,
    m_ailabs_en_uk_elizabeth_klett_speech_dataset,
    m_ailabs_en_us_elliot_miller_speech_dataset,
    m_ailabs_en_us_judy_bieber_speech_dataset,
    m_ailabs_en_us_mary_ann_speech_dataset,
)
from run.data._loader.old_wsl_datasets import OLD_WSL_DATASETS
from run.data._loader.utils import (
    DataLoader,
    conventional_dataset_loader,
    dataset_loader,
    get_non_speech_segments_and_cache,
    is_normalized_audio_file,
    maybe_normalize_audio_and_cache,
    normalize_audio,
    normalize_audio_suffix,
    read_audio,
)
from run.data._loader.wsl_datasets import (
    ADRIENNE_WALKER_HELLER,
    ALICIA_HARRIS,
    ALICIA_HARRIS__MANUAL_POST,
    BETH_CAMERON,
    BETH_CAMERON__CUSTOM,
    ELISE_RANDALL,
    FRANK_BONACQUISTI,
    GEORGE_DRAKE_JR,
    HANUMAN_WELCH,
    HEATHER_DOE,
    HILARY_NORIEGA,
    JACK_RUTKOWSKI,
    JACK_RUTKOWSKI__MANUAL_POST,
    JOHN_HUNERLACH__NARRATION,
    JOHN_HUNERLACH__RADIO,
    MARK_ATHERLAY,
    MEGAN_SINCLAIR,
    SAM_SCHOLL,
    SAM_SCHOLL__MANUAL_POST,
    STEVEN_WAHLBERG,
    SUSAN_MURPHY,
    WSL_DATASETS,
)

# TODO: Consider updating M-AILABS and LJSpeech to Google Storage, so that we can download
# and upload them faster. It'll also give us protection, if the datasets are deleted.


DATASETS = typing.cast(typing.Dict[Speaker, DataLoader], WSL_DATASETS.copy())
DATASETS[LINDA_JOHNSON] = lj_speech_dataset  # type: ignore
DATASETS[JUDY_BIEBER] = m_ailabs_en_us_judy_bieber_speech_dataset
DATASETS[MARY_ANN] = m_ailabs_en_us_mary_ann_speech_dataset
DATASETS[ELLIOT_MILLER] = m_ailabs_en_us_elliot_miller_speech_dataset
DATASETS[ELIZABETH_KLETT] = m_ailabs_en_uk_elizabeth_klett_speech_dataset


__all__ = [
    "data_structures",
    "m_ailabs",
    "utils",
    "Alignment",
    "NonalignmentSpans",
    "Passage",
    "Span",
    "Speaker",
    "alignment_dtype",
    "has_a_mistranscription",
    "voiced_nonalignment_spans",
    "m_ailabs_en_uk_elizabeth_klett_speech_dataset",
    "m_ailabs_en_us_elliot_miller_speech_dataset",
    "m_ailabs_en_us_judy_bieber_speech_dataset",
    "m_ailabs_en_us_mary_ann_speech_dataset",
    "conventional_dataset_loader",
    "dataset_loader",
    "get_non_speech_segments_and_cache",
    "is_normalized_audio_file",
    "maybe_normalize_audio_and_cache",
    "normalize_audio",
    "normalize_audio_suffix",
    "read_audio",
    "ADRIENNE_WALKER_HELLER",
    "ALICIA_HARRIS__MANUAL_POST",
    "ALICIA_HARRIS",
    "BETH_CAMERON__CUSTOM",
    "BETH_CAMERON",
    "ELISE_RANDALL",
    "ELIZABETH_KLETT",
    "ELLIOT_MILLER",
    "FRANK_BONACQUISTI",
    "GEORGE_DRAKE_JR",
    "HANUMAN_WELCH",
    "HEATHER_DOE",
    "HILARY_NORIEGA",
    "JACK_RUTKOWSKI__MANUAL_POST",
    "JACK_RUTKOWSKI",
    "JOHN_HUNERLACH__NARRATION",
    "JOHN_HUNERLACH__RADIO",
    "JUDY_BIEBER",
    "LINDA_JOHNSON",
    "MARK_ATHERLAY",
    "MARY_ANN",
    "MEGAN_SINCLAIR",
    "SAM_SCHOLL__MANUAL_POST",
    "SAM_SCHOLL",
    "STEVEN_WAHLBERG",
    "SUSAN_MURPHY",
    "DATASETS",
    "WSL_DATASETS",
    "OLD_WSL_DATASETS",
]

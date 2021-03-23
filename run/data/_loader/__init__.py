import typing
from pathlib import Path

from lib.datasets import m_ailabs
from lib.datasets.lj_speech import LINDA_JOHNSON, lj_speech_dataset
from lib.datasets.m_ailabs import (
    ELIZABETH_KLETT,
    ELLIOT_MILLER,
    JUDY_BIEBER,
    MARY_ANN,
    m_ailabs_en_uk_elizabeth_klett_speech_dataset,
    m_ailabs_en_us_elliot_miller_speech_dataset,
    m_ailabs_en_us_judy_bieber_speech_dataset,
    m_ailabs_en_us_mary_ann_speech_dataset,
)
from lib.datasets.old_wsl_datasets import OLD_WSL_DATASETS
from lib.datasets.utils import (
    Alignment,
    Passage,
    Span,
    SpanGenerator,
    Speaker,
    alignment_dtype,
    conventional_dataset_loader,
    dataset_loader,
    update_conventional_passage_script,
    update_passage_audio,
)
from lib.datasets.wsl_datasets import (
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


DataLoader = typing.Callable[[Path], typing.List[Passage]]
DATASETS = typing.cast(typing.Dict[Speaker, DataLoader], WSL_DATASETS.copy())
DATASETS[LINDA_JOHNSON] = lj_speech_dataset  # type: ignore
DATASETS[JUDY_BIEBER] = m_ailabs_en_us_judy_bieber_speech_dataset
DATASETS[MARY_ANN] = m_ailabs_en_us_mary_ann_speech_dataset
DATASETS[ELLIOT_MILLER] = m_ailabs_en_us_elliot_miller_speech_dataset
DATASETS[ELIZABETH_KLETT] = m_ailabs_en_uk_elizabeth_klett_speech_dataset


__all__ = [
    "Alignment",
    "IsConnected",
    "Passage",
    "Span",
    "SpanGenerator",
    "Speaker",
    "alignment_dtype",
    "conventional_dataset_loader",
    "dataset_loader",
    "update_conventional_passage_script",
    "update_passage_audio",
    "lj_speech_dataset",
    "m_ailabs",
    "m_ailabs_en_us_judy_bieber_speech_dataset",
    "m_ailabs_en_us_mary_ann_speech_dataset",
    "m_ailabs_en_us_elliot_miller_speech_dataset",
    "m_ailabs_en_uk_elizabeth_klett_speech_dataset",
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

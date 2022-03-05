import typing

from run.data._loader.data_structures import Speaker
from run.data._loader.english import m_ailabs, wsl, wsl_archive
from run.data._loader.english.m_ailabs import (
    ELIZABETH_KLETT,
    ELLIOT_MILLER,
    JUDY_BIEBER,
    MARY_ANN,
    m_ailabs_en_uk_elizabeth_klett_speech_dataset,
    m_ailabs_en_us_elliot_miller_speech_dataset,
    m_ailabs_en_us_judy_bieber_speech_dataset,
    m_ailabs_en_us_mary_ann_speech_dataset,
)
from run.data._loader.english.wsl import (
    ADRIENNE_WALKER_HELLER,
    ADRIENNE_WALKER_HELLER__PROMO,
    ALICIA_HARRIS,
    ALICIA_HARRIS__MANUAL_POST,
    ALISTAIR_DAVIS__EN_GB,
    BETH_CAMERON,
    BETH_CAMERON__CUSTOM,
    BRIAN_DIAMOND__EN_IE__PROMO,
    CHRISTOPHER_DANIELS__PROMO,
    DAMON_PAPADOPOULOS__PROMO,
    DAN_FURCA__PROMO,
    DANA_HURLEY__PROMO,
    DARBY_CUPIT__PROMO,
    ED_LACOMB__PROMO,
    ELISE_RANDALL,
    ENERGY_INDUSTRY_ACADEMY__CUSTOM_VOICE,
    FRANK_BONACQUISTI,
    GEORGE_DRAKE_JR,
    HANUMAN_WELCH,
    HAPPIFY__CUSTOM_VOICE,
    HEATHER_DOE,
    HILARY_NORIEGA,
    HOUR_ONE_NBC__BB_CUSTOM_VOICE,
    IZZY_TUGMAN__PROMO,
    JACK_RUTKOWSKI,
    JACK_RUTKOWSKI__MANUAL_POST,
    JOHN_HUNERLACH__NARRATION,
    JOHN_HUNERLACH__RADIO,
    LINSAY_ROUSSEAU__PROMO,
    MARI_MONGE__PROMO,
    MARK_ATHERLAY,
    MEGAN_SINCLAIR,
    NAOMI_MERCER_MCKELL__PROMO,
    OTIS_JIRY__STORY,
    SAM_SCHOLL,
    SAM_SCHOLL__MANUAL_POST,
    SAM_SCHOLL__PROMO,
    SHARON_GAULD_ALEXANDER__PROMO,
    SHAWN_WILLIAMS__PROMO,
    STEVEN_WAHLBERG,
    SUPER_HI_FI__CUSTOM_VOICE,
    SUSAN_MURPHY,
    THE_EXPLANATION_COMPANY__CUSTOM_VOICE,
    US_PHARMACOPEIA__CUSTOM_VOICE,
    VERITONE__CUSTOM_VOICE,
    VIACOM__CUSTOM_VOICE,
    WSL_DATASETS,
)
from run.data._loader.english.wsl_archive import (
    JOSIE__CUSTOM,
    JOSIE__CUSTOM__MANUAL_POST,
    LINCOLN__CUSTOM,
    OLD_WSL_DATASETS,
)
from run.data._loader.lj_speech import LINDA_JOHNSON, lj_speech_dataset
from run.data._loader.utils import DataLoader

# TODO: Consider updating M-AILABS and LJSpeech to Google Storage, so that we can download
# and upload them faster. It'll also give us protection, if the datasets are deleted.

DATASETS = typing.cast(typing.Dict[Speaker, DataLoader], WSL_DATASETS.copy())
DATASETS[LINDA_JOHNSON] = lj_speech_dataset  # type: ignore
DATASETS[JUDY_BIEBER] = m_ailabs_en_us_judy_bieber_speech_dataset
DATASETS[MARY_ANN] = m_ailabs_en_us_mary_ann_speech_dataset
DATASETS[ELLIOT_MILLER] = m_ailabs_en_us_elliot_miller_speech_dataset
DATASETS[ELIZABETH_KLETT] = m_ailabs_en_uk_elizabeth_klett_speech_dataset

__all__ = [
    "m_ailabs",
    "wsl",
    "wsl_archive",
    "ELIZABETH_KLETT",
    "ELLIOT_MILLER",
    "JUDY_BIEBER",
    "MARY_ANN",
    "m_ailabs_en_uk_elizabeth_klett_speech_dataset",
    "m_ailabs_en_us_elliot_miller_speech_dataset",
    "m_ailabs_en_us_judy_bieber_speech_dataset",
    "m_ailabs_en_us_mary_ann_speech_dataset",
    "JOSIE__CUSTOM",
    "JOSIE__CUSTOM__MANUAL_POST",
    "LINCOLN__CUSTOM",
    "OLD_WSL_DATASETS",
    "ADRIENNE_WALKER_HELLER",
    "ADRIENNE_WALKER_HELLER__PROMO",
    "ALICIA_HARRIS",
    "ALICIA_HARRIS__MANUAL_POST",
    "ALISTAIR_DAVIS__EN_GB",
    "BETH_CAMERON",
    "BETH_CAMERON__CUSTOM",
    "BRIAN_DIAMOND__EN_IE__PROMO",
    "CHRISTOPHER_DANIELS__PROMO",
    "DAMON_PAPADOPOULOS__PROMO",
    "DAN_FURCA__PROMO",
    "DANA_HURLEY__PROMO",
    "DARBY_CUPIT__PROMO",
    "ED_LACOMB__PROMO",
    "ELISE_RANDALL",
    "ENERGY_INDUSTRY_ACADEMY__CUSTOM_VOICE",
    "FRANK_BONACQUISTI",
    "GEORGE_DRAKE_JR",
    "HANUMAN_WELCH",
    "HAPPIFY__CUSTOM_VOICE",
    "HEATHER_DOE",
    "HILARY_NORIEGA",
    "HOUR_ONE_NBC__BB_CUSTOM_VOICE",
    "IZZY_TUGMAN__PROMO",
    "JACK_RUTKOWSKI",
    "JACK_RUTKOWSKI__MANUAL_POST",
    "JOHN_HUNERLACH__NARRATION",
    "JOHN_HUNERLACH__RADIO",
    "LINSAY_ROUSSEAU__PROMO",
    "MARI_MONGE__PROMO",
    "MARK_ATHERLAY",
    "MEGAN_SINCLAIR",
    "NAOMI_MERCER_MCKELL__PROMO",
    "OTIS_JIRY__STORY",
    "SAM_SCHOLL",
    "SAM_SCHOLL__MANUAL_POST",
    "SAM_SCHOLL__PROMO",
    "SHARON_GAULD_ALEXANDER__PROMO",
    "SHAWN_WILLIAMS__PROMO",
    "STEVEN_WAHLBERG",
    "SUPER_HI_FI__CUSTOM_VOICE",
    "SUSAN_MURPHY",
    "THE_EXPLANATION_COMPANY__CUSTOM_VOICE",
    "US_PHARMACOPEIA__CUSTOM_VOICE",
    "VERITONE__CUSTOM_VOICE",
    "VIACOM__CUSTOM_VOICE",
    "WSL_DATASETS",
    "DATASETS",
]
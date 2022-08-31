from functools import partial

from run.data._loader import structures
from run.data._loader.utils import wsl_gcs_dataset_loader

MITEL_GERMAN__CUSTOM_VOICE = structures.Speaker(
    "mitel__custom_voice__de_de",
    structures.Style.OTHER,
    structures.Dialect.DE_DE,
    "Mitel (German Custom Voice)",
    "mitel__custom_voice__de_de",
)
FIVE9_CUSTOM_VOICE__DE_DE = structures.Speaker(
    "five_nine__custom_voice__de_de",
    structures.Style.OTHER,
    structures.Dialect.DE_DE,
    "Five9 (German Custom Voice)",
    "five_nine__custom_voice__de_de",
)


_wsl_speakers = [s for s in locals().values() if isinstance(s, structures.Speaker)]
WSL_DATASETS = {
    s: partial(wsl_gcs_dataset_loader, speaker=s, prefix=str(s.language.value).lower())
    for s in _wsl_speakers
}

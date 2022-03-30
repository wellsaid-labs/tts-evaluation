from functools import partial

from run.data._loader import structures as struc
from run.data._loader.utils import wsl_gcs_dataset_loader

FIVE_NINE__CUSTOM_VOICE__PT_BR = struc.Speaker(
    "five_nine__custom_voice__pt_br",
    struc.Style.OTHER,
    struc.Dialect.PT_BR,
    "Five9 (Portuguese Custom Voice)",
    "five_nine__custom_voice__pt_br",
)


_wsl_speakers = [s for s in locals().values() if isinstance(s, struc.Speaker)]
WSL_DATASETS = {
    s: partial(wsl_gcs_dataset_loader, speaker=s, prefix=str(s.language.value).lower())
    for s in _wsl_speakers
}

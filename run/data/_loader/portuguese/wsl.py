from functools import partial

from run.data._loader.data_structures import make_pt_speaker
from run.data._loader.utils import Speaker, wsl_gcs_dataset_loader

FIVE_NINE__CUSTOM_VOICE__PT_BR = make_pt_speaker(
    "five_nine__custom_voice__pt_br", "Five9 (Portuguese Custom Voice)"
)


_wsl_speakers = [s for s in locals().values() if isinstance(s, Speaker)]
WSL_DATASETS = {
    s: partial(wsl_gcs_dataset_loader, speaker=s, prefix=str(s.language.value).lower())
    for s in _wsl_speakers
}

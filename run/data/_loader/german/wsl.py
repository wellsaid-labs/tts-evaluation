from functools import partial

from run.data._loader.data_structures import make_de_speaker
from run.data._loader.utils import Speaker, wsl_gcs_dataset_loader

MITEL_GERMAN__CUSTOM_VOICE = make_de_speaker(
    "mitel__custom_voice__de_de", "Mitel (German Custom Voice)"
)
FIVE9_CUSTOM_VOICE__DE_DE = make_de_speaker(
    "five_nine__custom_voice__de_de", "Five9 (German Custom Voice)"
)


_wsl_speakers = [s for s in locals().values() if isinstance(s, Speaker)]
WSL_DATASETS = {
    s: partial(wsl_gcs_dataset_loader, speaker=s, prefix=str(s.language.value).lower())
    for s in _wsl_speakers
}

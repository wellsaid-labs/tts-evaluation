from functools import partial

from run.data._loader.data_structures import make_german_speaker
from run.data._loader.utils import Speaker, wsl_gcs_dataset_loader

MITEL_GERMAN__CUSTOM_VOICE = make_german_speaker("Mitel - German", "Mitel (German Custom Voice)")


_wsl_speakers = [s for s in locals().values() if isinstance(s, Speaker)]
WSL_DATASETS = {
    s: partial(wsl_gcs_dataset_loader, speaker=s, prefix=str(s.language)) for s in _wsl_speakers
}

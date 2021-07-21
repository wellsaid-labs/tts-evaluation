import typing
from functools import partial
from pathlib import Path

from run.data._loader.data_structures import WSL_Languages
from run.data._loader.utils import Passage, Speaker, dataset_loader

##############
#   GERMAN   #
##############
MITEL_GERMAN__CUSTOM_VOICE = Speaker(
    "Mitel - German", "Mitel (German Custom Voice)", language=WSL_Languages.GERMAN
)


def _dataset_loader(directory: Path, speaker: Speaker, **kwargs) -> typing.List[Passage]:

    # TODO: Remove references to __manual_post if not part of international datasets
    # TODO: Update GCS path for international datasets, once GCS is organized

    manual_post_suffix = "__manual_post"
    suffix = manual_post_suffix if manual_post_suffix in speaker.label else ""
    label = speaker.label.replace(manual_post_suffix, "")
    kwargs = dict(recordings_directory_name="recordings" + suffix, **kwargs)
    gcs_path = f"gs://wellsaid_labs_datasets/{label}/processed"
    return dataset_loader(directory, label, gcs_path, speaker, **kwargs)


# TODO: Expand for other new languages as they are added here.
_wsl_speakers__german = [
    s for s in locals().values() if isinstance(s, Speaker) and s.language is WSL_Languages.GERMAN
]
WSL_DATASETS__GERMAN = {s: partial(_dataset_loader, speaker=s) for s in _wsl_speakers__german}

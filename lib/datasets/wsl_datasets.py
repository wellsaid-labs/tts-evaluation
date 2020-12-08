import typing
from functools import partial
from pathlib import Path

from lib.datasets.utils import Passage, Speaker, dataset_loader

# TODO: Consider not using the actors realnames in the codebase in an effort to protect their
# privacy.

ADRIENNE_WALKER_HELLER = Speaker("Adrienne Walker-Heller")
ALICIA_HARRIS = Speaker("Alicia Harris")
ALICIA_HARRIS__MANUAL_POST = Speaker("Alicia Harris (Manual Post Processing)")
BETH_CAMERON = Speaker("Beth Cameron")
BETH_CAMERON__CUSTOM = Speaker("Beth Cameron (Custom)")
ELISE_RANDALL = Speaker("Elise Randall")
FRANK_BONACQUISTI = Speaker("Frank Bonacquisti")
GEORGE_DRAKE_JR = Speaker("George Drake, Jr.")
HANUMAN_WELCH = Speaker("Hanuman Welch")
HEATHER_DOE = Speaker("Heather Doe")
HILARY_NORIEGA = Speaker("Hilary Noriega")
JACK_RUTKOWSKI = Speaker("Jack Rutkowski")
JACK_RUTKOWSKI__MANUAL_POST = Speaker("Jack Rutkowski (Manual Post Processing)")
JOHN_HUNERLACH__NARRATION = Speaker("John Hunerlach (Narration)")
JOHN_HUNERLACH__RADIO = Speaker("John Hunerlach (Radio)")
MARK_ATHERLAY = Speaker("Mark Atherlay")
MEGAN_SINCLAIR = Speaker("Megan Sinclair")
SAM_SCHOLL = Speaker("Sam Scholl")
SAM_SCHOLL__MANUAL_POST = Speaker("Sam Scholl (Manual Post Processing)")
STEVEN_WAHLBERG = Speaker("Steven Wahlberg")
SUSAN_MURPHY = Speaker("Susan Murphy")
_speaker_to_label = {v: k.lower() for k, v in locals().items() if isinstance(v, Speaker)}
_manual_post_suffix = "__manual_post"
_wsl_speakers = [s for s in locals().values() if isinstance(s, Speaker)]


def _dataset_loader(directory: Path, speaker: Speaker, **kwargs) -> typing.List[Passage]:
    label = _speaker_to_label[speaker]
    suffix = _manual_post_suffix if _manual_post_suffix in label else ""
    label = label.replace(_manual_post_suffix, "")
    kwargs = dict(recordings_directory_name="recordings" + suffix, **kwargs)
    gcs_path = f"gs://wellsaid_labs_datasets/{label}/processed"
    return dataset_loader(directory, label, gcs_path, speaker, **kwargs)


WSL_DATASETS = {s: partial(_dataset_loader, speaker=s) for s in _wsl_speakers}

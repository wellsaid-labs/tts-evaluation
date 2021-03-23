import typing
from functools import partial
from pathlib import Path

from run.data._loader.utils import Passage, Speaker, dataset_loader

# TODO: Consider not using the actors realnames in the codebase in an effort to protect their
# privacy.

ADRIENNE_WALKER_HELLER = Speaker("adrienne_walker_heller", "Adrienne Walker-Heller")
ALICIA_HARRIS = Speaker("alicia_harris", "Alicia Harris")
ALICIA_HARRIS__MANUAL_POST = Speaker(
    "alicia_harris__manual_post", "Alicia Harris (Manual Post Processing)"
)
BETH_CAMERON = Speaker("beth_cameron", "Beth Cameron")
BETH_CAMERON__CUSTOM = Speaker("beth_cameron__custom", "Beth Cameron (Custom)")
ELISE_RANDALL = Speaker("elise_randall", "Elise Randall")
FRANK_BONACQUISTI = Speaker("frank_bonacquisti", "Frank Bonacquisti")
GEORGE_DRAKE_JR = Speaker("george_drake_jr", "George Drake, Jr.")
HANUMAN_WELCH = Speaker("hanuman_welch", "Hanuman Welch")
HEATHER_DOE = Speaker("heather_doe", "Heather Doe")
HILARY_NORIEGA = Speaker("hilary_noriega", "Hilary Noriega")
JACK_RUTKOWSKI = Speaker("jack_rutkowski", "Jack Rutkowski")
JACK_RUTKOWSKI__MANUAL_POST = Speaker(
    "jack_rutkowski__manual_post", "Jack Rutkowski (Manual Post Processing)"
)
JOHN_HUNERLACH__NARRATION = Speaker("john_hunerlach__narration", "John Hunerlach (Narration)")
JOHN_HUNERLACH__RADIO = Speaker("john_hunerlach__radio", "John Hunerlach (Radio)")
MARK_ATHERLAY = Speaker("mark_atherlay", "Mark Atherlay")
MEGAN_SINCLAIR = Speaker("megan_sinclair", "Megan Sinclair")
SAM_SCHOLL = Speaker("sam_scholl", "Sam Scholl")
SAM_SCHOLL__MANUAL_POST = Speaker("sam_scholl__manual_post", "Sam Scholl (Manual Post Processing)")
STEVEN_WAHLBERG = Speaker("steven_wahlberg", "Steven Wahlberg")
SUSAN_MURPHY = Speaker("susan_murphy", "Susan Murphy")


def _dataset_loader(directory: Path, speaker: Speaker, **kwargs) -> typing.List[Passage]:
    manual_post_suffix = "__manual_post"
    suffix = manual_post_suffix if manual_post_suffix in speaker.label else ""
    label = speaker.label.replace(manual_post_suffix, "")
    kwargs = dict(recordings_directory_name="recordings" + suffix, **kwargs)
    gcs_path = f"gs://wellsaid_labs_datasets/{label}/processed"
    return dataset_loader(directory, label, gcs_path, speaker, **kwargs)


_wsl_speakers = [s for s in locals().values() if isinstance(s, Speaker)]
WSL_DATASETS = {s: partial(_dataset_loader, speaker=s) for s in _wsl_speakers}

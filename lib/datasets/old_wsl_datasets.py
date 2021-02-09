import logging
import typing
from functools import partial
from pathlib import Path

from torchnlp.download import download_file_maybe_extract

from lib.datasets.utils import Passage, Speaker, conventional_dataset_loader, make_passages
from lib.datasets.wsl_datasets import (
    ADRIENNE_WALKER_HELLER,
    ALICIA_HARRIS,
    BETH_CAMERON,
    BETH_CAMERON__CUSTOM,
    ELISE_RANDALL,
    FRANK_BONACQUISTI,
    GEORGE_DRAKE_JR,
    HANUMAN_WELCH,
    HEATHER_DOE,
    HILARY_NORIEGA,
    JACK_RUTKOWSKI,
    MARK_ATHERLAY,
    MEGAN_SINCLAIR,
    SAM_SCHOLL,
    STEVEN_WAHLBERG,
    SUSAN_MURPHY,
)

logger = logging.getLogger(__name__)

_metadata = {
    (
        "Hilary Noriega",
        HILARY_NORIEGA,
        "https://drive.google.com/uc?export=download&id=1VKefPVjDCfc1Qwb-gRHoGh0kyX__uOG8",
        "Hilary Noriega.tar.gz",
        False,
    ),
    (
        "Heather Doe",
        HEATHER_DOE,
        "https://drive.google.com/uc?export=download&id=1kqKGkyQq0lA32Rgos0WI9m-widz8g1HY",
        "Heather Doe.tar.gz",
        False,
    ),
    (
        "Beth Cameron (Custom)",
        BETH_CAMERON__CUSTOM,
        "https://drive.google.com/uc?export=download&id=1OJBAtSoaDzdlW9NWUR20F6HJ6U_BXBK2",
        "Beth Cameron (Custom).tar.gz",
        False,
    ),
    (
        "Beth Cameron",
        BETH_CAMERON,
        "https://drive.google.com/uc?export=download&id=1A-at3ZI1Aknbr5fVqlDM-rOl3A1It27W",
        "Beth Cameron.tar.gz",
        False,
    ),
    (
        "Sam Scholl",
        SAM_SCHOLL,
        "https://drive.google.com/uc?export=download&id=1AvAwYWgUC300l9VNUMeW1Kk0jUHGJxky",
        "Sean Hannity.tar.gz",
        False,
    ),
    (
        "Susan Murphy",
        SUSAN_MURPHY,
        "https://drive.google.com/uc?export=download&id=1oHCa6cKcYLQQcmER65ASzSTPFPzsg3JQ",
        "Susan Murphy.tar.gz",
        False,
    ),
    (
        "Adrienne Walker-Heller",
        ADRIENNE_WALKER_HELLER,
        "https://drive.google.com/uc?export=download&id=1MAypaxctTPlQw5zmYD02uId3ruuGenoW",
        "Adrienne Walker-Heller.tar.gz",
        False,
    ),
    (
        "Frank Bonacquisti",
        FRANK_BONACQUISTI,
        "https://drive.google.com/uc?export=download&id=1IJLADnQm6Cw8tLJNNqfmDefPj-aVjH9l",
        "Frank Bonacquisti.tar.gz",
        False,
    ),
    (
        "AliciaHarris",
        ALICIA_HARRIS,
        "https://drive.google.com/uc?export=download&id=1x2_XGTTqrwXjSYWRDfGRsoV0aSWDHr6G",
        "AliciaHarris.tar.gz",
        True,
    ),
    (
        "George Drake, Jr. ",
        GEORGE_DRAKE_JR,
        "https://drive.google.com/uc?export=download&id=1WkpmekXdgFN3dc42Oo_O2lPcsblFegHH",
        "George Drake.tar.gz",
        False,
    ),
    (
        "MeganSinclair",
        MEGAN_SINCLAIR,
        "https://drive.google.com/uc?export=download&id=1waUWeXvrgchFjeXMmfBs55obK9u6qr30",
        "MeganSinclair.tar.gz",
        True,
    ),
    (
        "EliseRandall",
        ELISE_RANDALL,
        "https://drive.google.com/uc?export=download&id=1-lbK0J2a9pr-G0NpyxZjcl8Jlz0lvgsc",
        "EliseRandall.tar.gz",
        True,
    ),
    (
        "Hanuman Welch",
        HANUMAN_WELCH,
        "https://drive.google.com/uc?export=download&id=1dU4USVsAd_0aZmjOVCvwmK2_mdQFratZ",
        "HanumanWelch.tar.gz",
        True,
    ),
    (
        "JackRutkowski",
        JACK_RUTKOWSKI,
        "https://drive.google.com/uc?export=download&id=1n5DhLuvK56Ge57R7maD7Rs4dXVBTBy3l",
        "JackRutkowski.tar.gz",
        True,
    ),
    (
        "MarkAtherlay",
        MARK_ATHERLAY,
        "https://drive.google.com/uc?export=download&id=1qi2nRASZXQlzwsfykoaWXtmR_MYFISC5",
        "Mark Atherlay.tar.gz",
        True,
    ),
    (
        "StevenWahlberg",
        STEVEN_WAHLBERG,
        "https://drive.google.com/uc?export=download&id=1osZFUK7_fcnw5zTrSVhGCb5WBZfnGYdT",
        "StevenWahlberg.tar.gz",
        True,
    ),
}


def _dataset_loader(
    directory: Path,
    extracted_name: str,
    speaker: Speaker,
    url: str,
    url_filename: str,
    create_root: bool,
    check_file: str = "{extracted_name}/metadata.csv",
    metadata_text_column="Content",
    metadata_audio_column="WAV Filename",
    metadata_kwargs={},
    audio_path_template: str = "{directory}/wavs/{file_name}",
) -> typing.List[Passage]:
    logger.info("Loading `%s` speech dataset", extracted_name)
    check_file = check_file.format(extracted_name=extracted_name)
    if create_root:
        (directory / extracted_name).mkdir(exist_ok=True)
    download_file_maybe_extract(
        url=url,
        directory=str((directory / extracted_name if create_root else directory).absolute()),
        check_files=[check_file],
        filename=url_filename,
    )
    passages = conventional_dataset_loader(
        directory / extracted_name,
        speaker,
        metadata_text_column=metadata_text_column,
        metadata_audio_column=metadata_audio_column,
        metadata_kwargs=metadata_kwargs,
        audio_path_template=audio_path_template,
    )
    return list(make_passages([passages]))


OLD_WSL_DATASETS = {
    speaker: partial(
        _dataset_loader,
        extracted_name=extracted_name,
        speaker=speaker,
        url=url,
        url_filename=url_filename,
        create_root=create_root,
    )
    for extracted_name, speaker, url, url_filename, create_root in _metadata
}


_deprecated_wsl_dataset_metadata = {
    (
        "Sean Hannity",
        Speaker("Sean Hannity"),
        "https://drive.google.com/uc?export=download&id=1YHX6yl1kX7lQguxSs4sJ1FPrAS9NZ8O4",
        "Sean Hannity.tar.gz",
        False,
    ),
    (
        "Nadine Nagamatsu",
        Speaker("Nadine Nagamatsu"),
        "https://drive.google.com/uc?export=download&id=1fwW6oV7x3QYImSfG811vhfjp8jKXVMGZ",
        "Nadine Nagamatsu.tar.gz",
        False,
    ),
    (
        "Lincoln_custom_ma",
        Speaker("Lincoln_custom_ma", "Lincoln (Custom)"),
        "https://drive.google.com/uc?export=download&id=1NJkVrPyxiNLKhc1Pj-ssCFhx_Mxzervf",
        "Lincoln_custom_ma.tar.gz",
        True,
    ),
}

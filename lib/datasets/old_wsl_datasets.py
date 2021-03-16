import logging
import typing
from functools import partial
from pathlib import Path

from third_party import LazyLoader
from torchnlp.download import download_file_maybe_extract

from lib.datasets.utils import Passage, Speaker, conventional_dataset_loader, make_passages
from lib.datasets.wsl_datasets import (
    ADRIENNE_WALKER_HELLER,
    ADRIENNE_WALKER_HELLER__PROMO,
    ALICIA_HARRIS,
    BETH_CAMERON,
    BETH_CAMERON__CUSTOM,
    DAMON_PAPADOPOULOS__PROMO,
    DANA_HURLEY__PROMO,
    ED_LACOMB__PROMO,
    ELISE_RANDALL,
    FRANK_BONACQUISTI,
    GEORGE_DRAKE_JR,
    HANUMAN_WELCH,
    HEATHER_DOE,
    HILARY_NORIEGA,
    JACK_RUTKOWSKI,
    JOHN_HUNERLACH__NARRATION,
    JOHN_HUNERLACH__RADIO,
    LINSAY_ROUSSEAU__PROMO,
    MARI_MONGE__PROMO,
    MARK_ATHERLAY,
    MEGAN_SINCLAIR,
    OTIS_JIRY__STORY,
    SAM_SCHOLL,
    SAM_SCHOLL__PROMO,
    STEVEN_WAHLBERG,
    SUSAN_MURPHY,
)

if typing.TYPE_CHECKING:  # pragma: no cover
    import pandas
else:
    pandas = LazyLoader("pandas", globals(), "pandas")

logger = logging.getLogger(__name__)

_metadata = {
    ##############
    # E-LEARNING #
    ##############
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
        "Sam Scholl.tar.gz",
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
        "john_hunerlach__narration",
        JOHN_HUNERLACH__NARRATION,
        "https://drive.google.com/uc?export=download&id=1-4BxZm6DdF20JmkUdkOdzoZG9j8oeALn",
        "john_hunerlach__narration.tar.gz",
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
    ###############
    # PROMOTIONAL #
    ###############
    (
        "AdrienneWalker__promo",
        ADRIENNE_WALKER_HELLER__PROMO,
        "https://drive.google.com/uc?export=download&id=113DYQm-Axr4CimuorhMLMLG_OUwJEgGr",
        "AdrienneWalker__promo.tar.gz",
        True,
    ),
    (
        "DamonPapadopoulos__promo",
        DAMON_PAPADOPOULOS__PROMO,
        "https://drive.google.com/uc?export=download&id=10T0_9AO967rx6gs-9wVB5r-9Uwe8vu_O",
        "DamonPapadopoulos__promo.tar.gz",
        True,
    ),
    (
        "DanaHurley__promo",
        DANA_HURLEY__PROMO,
        "https://drive.google.com/uc?export=download&id=10omgmnmoEjdB0YoIzPgSoClpj5s_ha40",
        "DanaHurley__promo.tar.gz",
        True,
    ),
    (
        "EdLaComb__promo",
        ED_LACOMB__PROMO,
        "https://drive.google.com/uc?export=download&id=10rR3JwUQIkOqczZ99JjJfSOYzwO0uUNM",
        "EdLaComb__promo.tar.gz",
        True,
    ),
    (
        "john_hunerlach__radio",
        JOHN_HUNERLACH__RADIO,
        "https://drive.google.com/uc?export=download&id=1ZTu_qrnY2DDkdRJ-Vx0LexaeoOXuH-Vh",
        "john_hunerlach__radio.tar.gz",
        True,
    ),
    (
        "LinsayRousseau__promo",
        LINSAY_ROUSSEAU__PROMO,
        "https://drive.google.com/uc?export=download&id=10uAc07lPmWdAEsijfI0kWaIiz0utDYJc",
        "LinsayRousseau__promo.tar.gz",
        True,
    ),
    (
        "SamScholl__promo",
        SAM_SCHOLL__PROMO,
        "https://drive.google.com/uc?export=download&id=112A7wEv61Mdcv8K0UW1FVYab_WdpRWCg",
        "SamScholl__promo.tar.gz",
        True,
    ),
    #########
    # OTHER #
    #########
    (
        "OtisJiry__promo",
        OTIS_JIRY__STORY,
        "https://drive.google.com/uc?export=download&id=11AQ_36XUkN3kodA8Lk7ucKrhPkUFDi55",
        "OtisJiry__promo.tar.gz",
        True,
    ),
    (
        "MariMonge__promo",
        MARI_MONGE__PROMO,
        "https://drive.google.com/uc?export=download&id=10xTRts6r01gDowM3yh6HOxMf679yDXYd",
        "MariMonge__promo.tar.gz",
        True,
    ),
}


def _dataset_loader(
    directory: Path,
    root_directory_name: str,
    speaker: Speaker,
    url: str,
    url_filename: str,
    create_root: bool,
    metadata_file_name: str = "metadata.csv",
    metadata_text_column="Content",
    metadata_audio_column="WAV Filename",
    metadata_kwargs={},
    audio_path_template: str = "{directory}/wavs/{file_name}",
    rename_template: str = "{speaker_label}__old",
) -> typing.List[Passage]:
    """Load an old WSL dataset.

    Learn more via the old data loader:
    https://github.com/wellsaid-labs/Text-to-Speech/blob/9b22a020b026bbe35fe3dfb30058effa43f2f3cb/src/datasets/utils.py#L240

    Args:
        ...
        root_directory_name: Name of the directory inside `directory` to store data. With
            `create_root=False`, this assumes the directory will be created while extracting `url`.
        ...
        url: URL of the dataset file.
        url_filename: Name of the file downloaded; Otherwise, a filename is extracted from the url.
        create_root: If `True` extract tar into `directory / root_directory_name` instead of
            `directory`.
        ...
        rename_template: A template specifying how to rename the top level folder.
    """
    logger.info("Loading `%s` speech dataset", root_directory_name)

    path = directory / root_directory_name
    new_path = directory / rename_template.format(speaker_label=speaker.label)
    if create_root and not new_path.exists():
        path.mkdir(exist_ok=True)

    if not new_path.exists():
        download_directory = str((path if create_root else directory).absolute())
        check_files = [str((path / metadata_file_name).absolute())]
        # TODO: Delete temporary `.tar.gz` files after extracting.
        download_file_maybe_extract(url, download_directory, url_filename, check_files=check_files)

    if path.exists():  # NOTE: Normalize file paths.
        path.rename(new_path)
        for item in list(new_path.glob("**/*")):
            normalized = item.parent / ((item.stem + item.suffix).replace(" ", "_"))
            if normalized != item:
                item.rename(normalized)
        df = pandas.read_csv(new_path / metadata_file_name, **metadata_kwargs)
        df[metadata_audio_column] = df[metadata_audio_column].apply(lambda f: f.replace(" ", "_"))
        df.to_csv(new_path / metadata_file_name, **metadata_kwargs)

    passages = conventional_dataset_loader(
        new_path,
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
        root_directory_name=root_directory_name,
        speaker=speaker,
        url=url,
        url_filename=url_filename,
        create_root=create_root,
    )
    for root_directory_name, speaker, url, url_filename, create_root in _metadata
}


_deprecated_metadata = {
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
    (
        "Josie_Custom",
        Speaker("Josie_Custom", "Josie (Custom)"),
        "https://drive.google.com/uc?export=download&id=1KPPjVMgCWCf-efkZBiCivbpiIt5z3LcG",
        "Josie_Custom.tar.gz",
        True,
    ),
    (
        "Josie_Custom_Loudnorm",
        Speaker("Josie_Custom_Loudnorm", "Josie (Custom, Loudness Standardized)"),
        "https://drive.google.com/uc?export=download&id=1CeLacT0Ys6jiroJPH0U8aO0GaKemg0vK",
        "Josie_Custom_Loudnorm.tar.gz",
        True,
    ),
}

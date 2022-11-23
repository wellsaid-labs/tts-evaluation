import logging
import pathlib
import subprocess
import typing

from torchnlp.download import _maybe_extract

from run.data._loader import structures as struc
from run.data._loader.utils import conventional_dataset_loader

logger = logging.getLogger(__name__)
GCP_SPEAKER = struc.Speaker(
    "en-US-Wavenet-D",
    struc.Style.DICT,
    struc.Dialect.EN_US,
    "Google Cloud Speaker",
    "gcp_speaker",
)
SESSION = struc.Session((GCP_SPEAKER, ""))


def dictionary_dataset(
    directory: pathlib.Path,
    root_directory_name: str = "gcp_pronunciation_dictionary",
    gcs_path: str = "gs://wellsaid_labs_datasets",
    file_name: str = "gcp_pronunciation_dictionary.tar.gz",
    session: struc.Session = SESSION,
    metadata_text_column: typing.Union[str, int] = 1,
    strict: bool = False,
    **kwargs,
) -> struc.UnprocessedDataset:
    """Load the a pronunciation dictionary dataset.

    Args:
        directory: Directory to cache the dataset.
        root_directory_name: Name of the extracted dataset directory.
        gcs_path
        session
        metadata_text_column
        add_tqdm
        strict: Use `gsutil` to validate the source files.
        **kwargs: Key word arguments passed to `conventional_dataset_loader`.
    """
    logger.info(f'Loading "{root_directory_name}" speech dataset...')
    directory = directory / root_directory_name
    if strict or not directory.exists():
        directory.mkdir(exist_ok=True)
        command = f"gsutil cp -n {gcs_path}/{file_name} {directory}/"
        subprocess.run(command.split(), check=True)
        _maybe_extract(directory / file_name, directory)
    passages = conventional_dataset_loader(
        directory,
        session[0],
        **kwargs,
        metadata_text_column=metadata_text_column,
        get_session=lambda *_, **__: session,
    )
    return [passages]

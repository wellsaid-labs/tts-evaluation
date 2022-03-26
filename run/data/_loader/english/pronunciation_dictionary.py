import logging
import pathlib
import subprocess
import typing

from run.data._loader.data_structures import Passage, Session, make_en_speaker
from run.data._loader.utils import conventional_dataset_loader, make_passages

logger = logging.getLogger(__name__)
GCP_SPEAKER = make_en_speaker("gcp_speaker")
SESSION = Session((GCP_SPEAKER, ""))


def pronunciation_dictionary_dataset(
    directory: pathlib.Path,
    root_directory_name: str = "gcp_pronunciation_dictionary",
    gcs_path: str = "gs://wellsaid_labs_datasets/",
    session: Session = SESSION,
    metadata_text_column: typing.Union[str, int] = 0,
    add_tqdm: bool = False,
    strict: bool = False,
    **kwargs,
) -> typing.List[Passage]:
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
        command = f"gsutil cp -n {gcs_path}/{directory.name}/ {directory}/"
        subprocess.run(command.split(), check=True)
    passages = conventional_dataset_loader(
        directory,
        session[0],
        **kwargs,
        metadata_text_column=metadata_text_column,
    )
    get_session = lambda *_, **__: session
    return list(make_passages(root_directory_name, [passages], add_tqdm, get_session))

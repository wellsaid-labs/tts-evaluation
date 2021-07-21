"""
Module to download and process the M-AILABS speech dataset
http://www.m-ailabs.bayern/en/the-mailabs-speech-dataset/.

Most of the data is based on LibriVox and Project Gutenberg. The training data consist of nearly
thousand hours of audio and the text-files in prepared format.

A transcription is provided for each clip. Clips vary in length from 1 to 20 seconds and have a
total length of approximately shown in the list (and in the respective info.txt-files) below.

The texts were published between 1884 and 1964, and are in the public domain. The audio was
recorded by the LibriVox project and is also in the public domain – except for Ukrainian.

M-AILABS directory structure follows this format:

    en_US/by_book/[Speaker Gender]/[Speaker Name]/[Book Title]

Within each book directory, there's metadata.csv file and a wavs directory following the
convention of the LJSpeech dataset.

The current implementation uses types for books, genders, speakers to allows robust error checking.
"""
import logging
import typing
from pathlib import Path

from torchnlp.download import download_file_maybe_extract

from run.data._loader.data_structures import Passage, Session, Speaker, UnprocessedPassage
from run.data._loader.utils import conventional_dataset_loader, make_passages

logger = logging.getLogger(__name__)
Dataset = typing.NewType("Dataset", str)


class Book(typing.NamedTuple):
    dataset: Dataset
    speaker: Speaker
    title: str


def _book_to_metadata_path(book: Book, root: Path) -> Path:
    """Given a book of `Book` type, returns the relative path to its "metadata.csv" file."""
    assert book.speaker.gender is not None
    gender = book.speaker.gender.lower()
    return root / "by_book" / gender / book.speaker.label / book.title / "metadata.csv"


def _metadata_path_to_book(metadata_path: Path, root: Path) -> Book:
    """Given a path to a book's "metadata.csv", returns the corresponding `Book` object."""
    # EXAMPLE: "by_book/female/judy_bieber/dorothy_and_wizard_oz/metadata.csv"
    metadata_path = metadata_path.relative_to(root)
    speaker_gender, speaker_label, book_title = metadata_path.parts[1:4]
    speaker = Speaker(speaker_label, gender=speaker_gender.lower())
    return Book(Dataset(root.name), speaker, book_title)


def _get_session(passage: UnprocessedPassage) -> Session:
    """For the M-AILABS speech dataset, we define each chapter as an individual recording
    session."""
    chapter = passage.audio_path.stem.rsplit("_", 1)[0]
    label = f"{passage.audio_path.parent.parent.name}/{passage.audio_path.parent.name}/{chapter}"
    return Session(label)


def m_ailabs_speech_dataset(
    directory: Path,
    extracted_name: str,
    url: str,
    books: typing.List[Book],
    check_files: typing.List[str],
    root_directory_name: str = "M-AILABS",
    metadata_pattern: str = "**/metadata.csv",
    add_tqdm: bool = False,
    get_session: typing.Callable[[UnprocessedPassage], Session] = _get_session,
) -> typing.List[Passage]:
    """Download, extract, and process a M-AILABS dataset.

    NOTE: The original URL is `https://data.solak.de/2019/01/the-m-ailabs-speech-dataset/`. Use
    `curl -I <URL>` to find the redirected URL.

    Args:
        directory: Directory to cache the dataset.
        extracted_name: Name of the extracted dataset directory.
        url: URL of the dataset `tar.gz` file.
        books: List of books to load.
        check_files
        root_directory_name: Name of the dataset directory.
        metadata_pattern: Pattern for all `metadata.csv` files containing (filename, text)
            information.
        get_session
        add_tqdm
    """
    name = f"{root_directory_name} {extracted_name}"
    logger.info(f"Loading {name} speech dataset...")
    directory = directory / root_directory_name
    directory.mkdir(exist_ok=True)
    download_file_maybe_extract(url=url, directory=str(directory), check_files=check_files)
    directory = directory / extracted_name

    metadata_paths = list(directory.glob(metadata_pattern))
    downloaded_books = set([_metadata_path_to_book(p, directory) for p in metadata_paths])
    assert len(set(books) - downloaded_books) == 0, "Unable to find every book in `books`."

    passages: typing.List[typing.List[UnprocessedPassage]] = []
    for book in books:
        metadata_path = _book_to_metadata_path(book, directory)
        loaded = conventional_dataset_loader(
            metadata_path.parent, book.speaker, additional_metadata={"book": book}
        )
        passages.append(loaded)
    return list(make_passages(name, passages, add_tqdm, get_session))

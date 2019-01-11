from functools import partial
from pathlib import Path

import logging

import pandas

from src.datasets.constants import Speaker
from src.datasets.constants import TextSpeechRow
from src.datasets.process import _download_file_maybe_extract
from src.datasets.process import _normalize_audio_and_cache
from src.datasets.process import _process_in_parallel
from src.datasets.process import _split_dataset
from src.hparams import configurable

logger = logging.getLogger(__name__)


def _processing_row(row,
                    directory,
                    extracted_name,
                    kwargs,
                    metadata_text_column='Content',
                    metadata_audio_column='WAV Filename',
                    metadata_source_column='Source',
                    metadata_title_column='Title',
                    audio_directory='wavs',
                    speaker=Speaker.HILARY_NORIEGA):  # pragma: no cover
    """
    Note:
        - ``# pragma: no cover`` is used because this functionality is run with multiprocessing
          that is not compatible with the coverage module.

    Args:
        directory (str or Path, optional): Directory to cache the dataset.
        extracted_name (str, optional): Name of the extracted dataset directory.
        kwargs: Arguments passed to process dataset audio.
        metadata_text_column (str, optional): Column name or index with the audio transcript.
        metadata_audio_column (str, optional): Column name or index with the audio filename.
        metadata_source_column (str, optional): Column name or index with the source of the original
            script.
        metadata_title_column (str, optional): Column name or index with the title of the original
            script.
        audio_directory (str, optional): Name of the directory harboring audio files.
        speaker (src.datasets.Speaker, optional)

    Returns:
        (TextSpeechRow) Processed row.
    """
    text = row[metadata_text_column].strip()
    audio_path = Path(directory, extracted_name, audio_directory, row[metadata_audio_column])
    audio_path = _normalize_audio_and_cache(audio_path, **kwargs)
    return TextSpeechRow(
        text=text,
        audio_path=audio_path,
        speaker=speaker,
        metadata={
            'script_title': row[metadata_title_column],
            'script_source': row[metadata_source_column],
        })


@configurable
def hilary_dataset(
        directory='data/',
        extracted_name='Hilary',
        url='https://drive.google.com/uc?export=download&id=10rOAnbV_wslhvTvRnxNMc9aqWmk1NtYK',
        url_filename='Hilary.tar.gz',
        check_files=['Hilary/metadata.csv'],
        metadata_file='metadata.csv',
        metadata_delimiter=',',
        splits=(.8, .2),
        **kwargs):
    """ Load the Hilary Speech dataset by WellSaid.

    Args:
        directory (str or Path, optional): Directory to cache the dataset.
        extracted_name (str, optional): Name of the extracted dataset directory.
        url (str, optional): URL of the dataset `tar.gz` file.
        url_filename (str, optional): Filename of the downloaded file.
        check_files (list of str, optional): Check this file exists if the download was successful.
        metadata_file (str, optional): The file containing audio metadata.
        metadata_delimiter (str, optional): Delimiter for the metadata file.
        splits (tuple, optional): The number of splits and cardinality of dataset splits.
        **kwargs: Arguments passed to process dataset audio.

    Returns:
        list: Dataset with audio filenames and text annotations.

    Example:
        >>> from src.hparams import set_hparams # doctest: +SKIP
        >>> from src.datasets import hilary_dataset # doctest: +SKIP
        >>> set_hparams() # doctest: +SKIP
        >>> train, dev = hilary_dataset() # doctest: +SKIP
    """
    logger.info('Loading Hilary speech dataset')
    _download_file_maybe_extract(
        url=url, directory=str(directory), check_files=check_files, filename=url_filename)
    metadata_path = Path(directory, extracted_name, metadata_file)
    data = [
        row.to_dict()
        for _, row in pandas.read_csv(metadata_path, delimiter=metadata_delimiter).iterrows()
    ]
    logger.info('Normalizing text and audio...')
    data = _process_in_parallel(
        data,
        partial(
            _processing_row,
            directory=directory,
            extracted_name=extracted_name,
            kwargs=kwargs,
        ))
    return _split_dataset(data, splits=splits)

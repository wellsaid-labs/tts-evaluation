from pathlib import Path

import pandas

from torchnlp.download import download_file_maybe_extract

from src.datasets._process import process_all
from src.datasets._process import process_audio
from src.utils.configurable import configurable


@configurable
def hillary_dataset(
        directory='data/',
        extracted_name='Hillary',
        url='https://drive.google.com/uc?export=download&id=10rOAnbV_wslhvTvRnxNMc9aqWmk1NtYK',
        url_filename='Hillary.tar.gz',
        check_files=['Hillary/metadata.csv'],
        metadata_file='metadata.csv',
        text_column='Content',
        audio_column='WAV Filename',
        source_column='Source',
        title_column='Title',
        delimiter=',',
        audio_directory='wavs',
        random_seed=123,
        splits=(.8, .2),
        **kwargs):
    """ Load the Hillary Speech dataset by WellSaid.

    Args:
        directory (str or Path, optional): Directory to cache the dataset.
        extracted_name (str, optional): Name of the extracted dataset directory.
        url (str, optional): URL of the dataset `tar.gz` file.
        url_filename (str, optional): Filename of the downloaded file.
        check_files (list of str, optional): Check this file exists if the download was successful.
        metadata_file (str, optional): The file containing audio metadata.
        text_column (str, optional): Column name or index with the audio transcript.
        audio_column (str, optional): Column name or index with the audio filename.
        source_column (str, optional): Column name or index with the source of the original script.
        title_column (str, optional): Column name or index with the title of the original script.
        delimiter (str, optional): Delimiter for the metadata file.
        audio_directory (str, optional): Name of the directory harboring audio files.
        random_seed (int, optional): Random seed used to determine the splits.
        splits (tuple, optional): The number of splits and cardinality of dataset splits.
        **kwargs: Arguments passed to process dataset audio.

    Returns:
        :class:`torchnlp.datasets.Dataset`: Dataset with audio filenames and text annotations.

    Example:
        >>> import pprint # doctest: +SKIP
        >>> from src.datasets import hillary_dataset # doctest: +SKIP
        >>> train, dev = hillary_dataset() # doctest: +SKIP
        >>> pprint.pprint(train[0:2], width=80) # doctest: +SKIP
        [{'text': 'associated with the performance of "social" and "mechanical" tasks.',
          'audio_filename': PosixPath('data/Hillary/wavs/Scripts 16-21/'
                                    'script_86_chunk_15-rate=24000.wav')},
         {'text': 'Take an in-depth look at the American Red Cross history,',
          'audio_filename': PosixPath('data/Hillary/wavs/Scripts 34-39/'
                                    'script_58_chunk_3-rate=24000.wav')}]
    """
    download_file_maybe_extract(
        url=url, directory=str(directory), check_files=check_files, filename=url_filename)
    metadata_path = Path(directory, extracted_name, metadata_file)

    def extract_fun(row):
        row = row[1]
        audio_filename = Path(directory, extracted_name, audio_directory, getattr(
            row, audio_column))
        processed_audio_filename = process_audio(audio_filename, **kwargs)
        return {
            'text': getattr(row, text_column).strip(),
            'audio_filename': processed_audio_filename,
            'script_title': getattr(row, title_column),
            'script_source': getattr(row, source_column)
        }

    data_frame = pandas.read_csv(metadata_path, delimiter=delimiter).iterrows()
    return process_all(extract_fun, data_frame, splits, random_seed)

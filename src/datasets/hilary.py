from functools import partial
from pathlib import Path

import pandas

from torchnlp.datasets import Dataset
from torchnlp.download import download_file_maybe_extract

from src.datasets.constants import Speaker
from src.datasets.process import compute_spectrogram
from src.datasets.process import normalize_audio
from src.datasets.process import process_with_processes
from src.datasets.process import split_dataset
from src.utils import Checkpoint
from src.hparams import configurable


def _processing_func(row,
                     directory,
                     extracted_name,
                     spectrogram_model_checkpoint_path,
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
        spectrogram_model_checkpoint_path (str or None, optional): Spectrogram model to predict a
            ground truth aligned spectrogram.
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
        {
            text (str)
            script_title (str): The title of the script this text snippet is from.
            script_source (str): The source of the script this text snippet is from.
            audio_path (Path)
            spectrogram_path (Path)
            predicted_spectrogram_path (Path)
            speaker (src.datasets.Speaker)
        }
    """
    text = row[metadata_text_column].strip()
    audio_path = Path(directory, extracted_name, audio_directory, row[metadata_audio_column])
    audio_path = normalize_audio(audio_path, **kwargs)
    spectrogram_model_checkpoint = Checkpoint.from_path(spectrogram_model_checkpoint_path)
    audio_path, spectrogram_path, predicted_spectrogram_path = compute_spectrogram(
        audio_path, text, speaker, spectrogram_model_checkpoint)

    return {
        'text': text,
        'audio_path': audio_path,
        'script_title': row[metadata_title_column],
        'script_source': row[metadata_source_column],
        'spectrogram_path': spectrogram_path,
        'predicted_spectrogram_path': predicted_spectrogram_path,
        'speaker': speaker
    }


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
        spectrogram_model_checkpoint_path=None,
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
        spectrogram_model_checkpoint_path (str or None, optional): Spectrogram model to predict a
            ground truth aligned spectrogram.
        **kwargs: Arguments passed to process dataset audio.

    Returns:
        :class:`torchnlp.datasets.Dataset`: Dataset with audio filenames and text annotations.

    Example:
        >>> import pprint # doctest: +SKIP
        >>> from src.hparams import set_hparams # doctest: +SKIP
        >>> from src.datasets import hilary_dataset # doctest: +SKIP
        >>> set_hparams() # doctest: +SKIP
        >>> train, dev = hilary_dataset() # doctest: +SKIP
        >>> pprint.pprint(train[0:2], width=80) # doctest: +SKIP
        [{'audio_path': PosixPath('data/Hilary/wavs/Scripts 16-21/spectrogram(rate('
                                  'guard(script_86_chunk_15),24000)).npy'),
          'predicted_spectrogram_path': None,
          'script_source': 'Wikipedia',
          'script_title': 'Empathy: Neurological basis',
          'speaker': <src.datasets.constants.Speaker object at 0x11f54a470>,
          'spectrogram_path': PosixPath('data/Hilary/wavs/Scripts 16-21/pad(rate(guard('
                                        'script_86_chunk_15),24000)).npy'),
          'text': 'associated with the performance of "social" and "mechanical" tasks.'},
        {'audio_path': PosixPath('data/Hilary/wavs/Scripts 34-39/pad(rate('
                                  'guard(script_58_chunk_3),24000)).npy'),
          'predicted_spectrogram_path': None,
          'script_source': 'Edge Studio',
          'script_title': 'Red Cross',
          'speaker': <src.datasets.constants.Speaker object at 0x11f91af28>,
          'spectrogram_path': PosixPath('data/Hilary/wavs/Scripts 34-39/spectrogram(rate('
                                        'guard(script_58_chunk_3),24000)).npy'),
          'text': 'Take an in-depth look at the American Red Cross history,'}]
    """
    download_file_maybe_extract(
        url=url, directory=str(directory), check_files=check_files, filename=url_filename)
    metadata_path = Path(directory, extracted_name, metadata_file)

    data = [
        row.to_dict()
        for _, row in pandas.read_csv(metadata_path, delimiter=metadata_delimiter).iterrows()
    ]
    data = process_with_processes(
        data,
        partial(
            _processing_func,
            directory=directory,
            extracted_name=extracted_name,
            spectrogram_model_checkpoint_path=spectrogram_model_checkpoint_path,
            kwargs=kwargs,
        ))
    splits = split_dataset(data, splits=splits)
    return tuple(Dataset(split) for split in splits)

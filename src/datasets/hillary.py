from pathlib import Path

from torchnlp.download import download_file_maybe_extract

from src.datasets._process import process_all
from src.datasets._process import process_audio
from src.datasets._process import read_speech_data
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
        delimiter=',',
        header=True,
        resample=24000,
        norm=False,
        guard=False,
        random_seed=123,
        splits=(.8, .2),
        check_wavfiles=True):
    """ Load the Hillary Speech dataset by WellSaid.

    Args:
        directory (str or Path, optional): Directory to cache the dataset.
        extracted_name (str, optional): Name of the extracted dataset directory.
        url (str, optional): URL of the dataset `tar.gz` file.
        url_filename (str, optional): Filename of the downloaded file.
        check_files (list of str, optional): Check this file exists if the download was successful.
        metadata_file (str, optional): The file containing audio metadata.
        text_column (str or int, optional): Column name or index with the audio transcript.
        audio_column (str or int, optional): Column name or index with the audio filename.
        delimiter (str, optional): Delimiter for the metadata file.
        header (bool, optional): If True, ``metadata_file`` has a header to parse.
        resample (int or None, optional): If integer is provided, uses SoX to create resampled
            files.
        norm (bool, optional): Automatically invoke the gain effect to guard against clipping and to
            normalise the audio.
        guard (bool, optional): Automatically invoke the gain effect to guard against clipping.
        random_seed (int, optional): Random seed used to determine the splits.
        splits (tuple, optional): The number of splits and cardinality of dataset splits.
        check_wavfiles: If False, skip the check for existence of wav files.

    Returns:
        :class:`torchnlp.datasets.Dataset`: Dataset with audio filenames and text annotations.

    Example:
        >>> import pprint # doctest: +SKIP
        >>> from src.datasets import hillary_dataset # doctest: +SKIP
        >>> train, dev = hillary_dataset() # doctest: +SKIP
        >>> pprint.pprint(train[0:2], width=80) # doctest: +SKIP
        [{'text': 'associated with the performance of "social" and "mechanical" tasks.',
          'wav_filename': PosixPath('data/Hillary/wavs/Scripts 16-21/'
                                    'script_86_chunk_15-rate=24000.wav')},
         {'text': 'Take an in-depth look at the American Red Cross history,',
          'wav_filename': PosixPath('data/Hillary/wavs/Scripts 34-39/'
                                    'script_58_chunk_3-rate=24000.wav')}]
    """
    download_file_maybe_extract(
        url=url, directory=str(directory), check_files=check_files, filename=url_filename)
    path = Path(directory, extracted_name, metadata_file)

    def extract_fun(args):
        text, wav_filename = args
        processed_wav_filename = process_audio(
            wav_filename, resample=resample, norm=norm, guard=guard)
        return {'text': text, 'wav_filename': processed_wav_filename}

    data = read_speech_data(
        path,
        check_wavfiles=check_wavfiles,
        text_column=text_column,
        audio_column=audio_column,
        delimiter=delimiter,
        header=header)
    return process_all(extract_fun, data, splits, random_seed, check_wavfiles=check_wavfiles)

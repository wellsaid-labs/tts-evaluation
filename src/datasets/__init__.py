"""
TODO: Support more datasets:
  - English Bible Speech Dataset
    https://www.kaggle.com/bryanpark/the-world-english-bible-speech-dataset
  -	CMU Arctic Speech Synthesis Dataset
    http://festvox.org/cmu_arctic/
  - Synthetic Wave Dataset with SoX e.g. [sine, square, triangle, sawtooth, trapezium, exp, brow]
  - VCTK Dataset
  - Voice Conversion Challenge (VCC) 2016 dataset
  - Blizzard dataset
  - JSUT dataset
  - Common Voice dataset
    https://toolbox.google.com/datasetsearch/search?query=text%20speech&docid=sGZ%2FjOYUalNI7AzSAAAAAA%3D%3D
"""
from src.datasets.constants import Gender
from src.datasets.constants import Speaker
from src.datasets.constants import TextSpeechRow
from src.datasets.lj_speech import lj_speech_dataset
from src.datasets.m_ailabs import m_ailabs_speech_dataset
from src.datasets.utils import _dataset_loader
from src.datasets.utils import add_predicted_spectrogram_column
from src.datasets.utils import add_spectrogram_column


# TODO: Consider not using the actors realnames in the codebase in an effort to protect their
# privacy.
# TODO: Fix URL for Hilary
# TODO: Test every dataset
def hilary_speech_dataset(
        extracted_name='Hilary Noriega',
        url='https://drive.google.com/uc?export=download&id=10rOAnbV_wslhvTvRnxNMc9aqWmk1NtYK',
        url_filename='Hilary Noriega.zip',
        speaker=Speaker('Hilary Noriega', Gender.FEMALE),
        **kwargs):
    return _dataset_loader(extracted_name, url, speaker, url_filename, **kwargs)


def heather_speech_dataset(extracted_name='Heather Doe',
                           url='',
                           url_filename='Heather Doe.zip',
                           speaker=Speaker('Heather Doe', Gender.FEMALE),
                           **kwargs):
    return _dataset_loader(extracted_name, url, speaker, url_filename, **kwargs)


def beth_custom_speech_dataset(extracted_name='Beth Cameron (Custom)',
                               url='',
                               url_filename='Beth Cameron (Custom).zip',
                               speaker=Speaker('Beth Cameron', Gender.FEMALE),
                               **kwargs):
    return _dataset_loader(extracted_name, url, speaker, url_filename, **kwargs)


def beth_speech_dataset(extracted_name='Beth Cameron',
                        url='',
                        url_filename='Beth Cameron.zip',
                        speaker=Speaker('Beth Cameron', Gender.FEMALE),
                        **kwargs):
    return _dataset_loader(extracted_name, url, speaker, url_filename, **kwargs)


def sam_speech_dataset(extracted_name='Sam Scholl',
                       url='',
                       url_filename='Sam Scholl.zip',
                       speaker=Speaker('Sam Scholl', Gender.MALE),
                       **kwargs):
    return _dataset_loader(extracted_name, url, speaker, url_filename, **kwargs)


def susan_speech_dataset(extracted_name='Susan Murphy',
                         url='',
                         url_filename='Susan Murphy.zip',
                         speaker=Speaker('Susan Murphy', Gender.FEMALE),
                         **kwargs):
    return _dataset_loader(extracted_name, url, speaker, url_filename, **kwargs)


__all__ = [
    'Speaker', 'Gender', 'lj_speech_dataset', 'm_ailabs_speech_dataset', 'hilary_speech_dataset',
    'heather_speech_dataset', 'beth_speech_dataset', 'beth_custom_speech_dataset',
    'sam_speech_dataset', 'susan_speech_dataset', 'TextSpeechRow',
    'add_predicted_spectrogram_column', 'add_spectrogram_column'
]

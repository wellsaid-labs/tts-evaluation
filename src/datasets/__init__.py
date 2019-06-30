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
from src.datasets.m_ailabs import m_ailabs_en_uk_speech_dataset
from src.datasets.m_ailabs import m_ailabs_en_us_speech_dataset
from src.datasets.utils import _dataset_loader
from src.datasets.utils import add_predicted_spectrogram_column
from src.datasets.utils import add_spectrogram_column
from src.datasets.utils import filter_
from src.datasets.utils import normalize_audio_column

# TODO: Consider not using the actors realnames in the codebase in an effort to protect their
# privacy.
# TODO: Consider not using public Google Drive links for safety reasons.


def hilary_speech_dataset(
        extracted_name='Hilary Noriega',
        url='https://drive.google.com/uc?export=download&id=1VKefPVjDCfc1Qwb-gRHoGh0kyX__uOG8',
        url_filename='Hilary Noriega.tar.gz',
        speaker=Speaker('Hilary Noriega', Gender.FEMALE),
        **kwargs):
    return _dataset_loader(extracted_name, url, speaker, url_filename, **kwargs)


def heather_speech_dataset(
        extracted_name='Heather Doe',
        url='https://drive.google.com/uc?export=download&id=1kqKGkyQq0lA32Rgos0WI9m-widz8g1HY',
        url_filename='Heather Doe.tar.gz',
        speaker=Speaker('Heather Doe', Gender.FEMALE),
        **kwargs):
    return _dataset_loader(extracted_name, url, speaker, url_filename, **kwargs)


def beth_custom_speech_dataset(
        extracted_name='Beth Cameron (Custom)',
        url='https://drive.google.com/uc?export=download&id=1OJBAtSoaDzdlW9NWUR20F6HJ6U_BXBK2',
        url_filename='Beth Cameron (Custom).tar.gz',
        speaker=Speaker('Beth Cameron', Gender.FEMALE),
        **kwargs):
    return _dataset_loader(extracted_name, url, speaker, url_filename, **kwargs)


def beth_speech_dataset(
        extracted_name='Beth Cameron',
        url='https://drive.google.com/uc?export=download&id=1A-at3ZI1Aknbr5fVqlDM-rOl3A1It27W',
        url_filename='Beth Cameron.tar.gz',
        speaker=Speaker('Beth Cameron', Gender.FEMALE),
        **kwargs):
    return _dataset_loader(extracted_name, url, speaker, url_filename, **kwargs)


def sean_speech_dataset(
        extracted_name='Sean Hannity',
        url='https://drive.google.com/uc?export=download&id=1Qly0EIpkANQqQWWjsOI7Pai5rWdrInqT',
        url_filename='Sean Hannity.tar.gz',
        speaker=Speaker('Sean Hannity', Gender.MALE),
        **kwargs):
    return _dataset_loader(extracted_name, url, speaker, url_filename, **kwargs)


def sam_speech_dataset(
        extracted_name='Sam Scholl',
        url='https://drive.google.com/uc?export=download&id=1AvAwYWgUC300l9VNUMeW1Kk0jUHGJxky',
        url_filename='Sam Scholl.tar.gz',
        speaker=Speaker('Sam Scholl', Gender.MALE),
        **kwargs):
    return _dataset_loader(extracted_name, url, speaker, url_filename, **kwargs)


def susan_speech_dataset(
        extracted_name='Susan Murphy',
        url='https://drive.google.com/uc?export=download&id=1oHCa6cKcYLQQcmER65ASzSTPFPzsg3JQ',
        url_filename='Susan Murphy.tar.gz',
        speaker=Speaker('Susan Murphy', Gender.FEMALE),
        **kwargs):
    return _dataset_loader(extracted_name, url, speaker, url_filename, **kwargs)


def adrienne_speech_dataset(
        extracted_name='Adrienne Walker-Heller',
        url='https://drive.google.com/uc?export=download&id=1MAypaxctTPlQw5zmYD02uId3ruuGenoW',
        url_filename='Adrienne Walker-Heller.tar.gz',
        speaker=Speaker('Adrienne Walker-Heller', Gender.FEMALE),
        **kwargs):
    return _dataset_loader(extracted_name, url, speaker, url_filename, **kwargs)


def frank_speech_dataset(
        extracted_name='Frank Bonacquisti',
        url='https://drive.google.com/uc?export=download&id=1IJLADnQm6Cw8tLJNNqfmDefPj-aVjH9l',
        url_filename='Frank Bonacquisti.tar.gz',
        speaker=Speaker('Frank Bonacquisti', Gender.MALE),
        **kwargs):
    return _dataset_loader(extracted_name, url, speaker, url_filename, **kwargs)


__all__ = [
    'Speaker', 'Gender', 'lj_speech_dataset', 'm_ailabs_en_us_speech_dataset',
    'm_ailabs_en_uk_speech_dataset', 'hilary_speech_dataset', 'heather_speech_dataset',
    'beth_speech_dataset', 'beth_custom_speech_dataset', 'sam_speech_dataset',
    'susan_speech_dataset', 'sean_speech_dataset', 'TextSpeechRow',
    'add_predicted_spectrogram_column', 'add_spectrogram_column', 'filter_',
    'normalize_audio_column'
]

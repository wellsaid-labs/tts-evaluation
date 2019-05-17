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
from src.datasets.utils import filter_
from src.datasets.utils import normalize_audio_column


# TODO: Consider not using the actors realnames in the codebase in an effort to protect their
# privacy.
# TODO: Consider not using public Google Drive links for safety reasons.
def hilary_speech_dataset(
        extracted_name='Hilary Noriega',
        url='https://drive.google.com/uc?export=download&id=18nU0L0gFDVU65ViVs9s1yh1aXEPH1kEI',
        url_filename='Hilary Noriega.tar.gz',
        speaker=Speaker('Hilary Noriega', Gender.FEMALE),
        **kwargs):
    return _dataset_loader(extracted_name, url, speaker, url_filename, **kwargs)


def heather_speech_dataset(
        extracted_name='Heather Doe',
        url='https://drive.google.com/uc?export=download&id=18LSE4jvB7eviZM9I7xEJC_OVjezfsx1Y',
        url_filename='Heather Doe.tar.gz',
        speaker=Speaker('Heather Doe', Gender.FEMALE),
        **kwargs):
    return _dataset_loader(extracted_name, url, speaker, url_filename, **kwargs)


def beth_custom_speech_dataset(
        extracted_name='Beth Cameron (Custom)',
        url='https://drive.google.com/uc?export=download&id=13GiBkzaQNxMYtxsvc26Us8ubOldl6aF_',
        url_filename='Beth Cameron (Custom).tar.gz',
        speaker=Speaker('Beth Cameron', Gender.FEMALE),
        **kwargs):
    return _dataset_loader(extracted_name, url, speaker, url_filename, **kwargs)


def beth_speech_dataset(
        extracted_name='Beth Cameron',
        url='https://drive.google.com/uc?export=download&id=1WF7E1H9vnRIQTLZM6AFyW-G-HfNMTUvx',
        url_filename='Beth Cameron.tar.gz',
        speaker=Speaker('Beth Cameron', Gender.FEMALE),
        **kwargs):
    return _dataset_loader(extracted_name, url, speaker, url_filename, **kwargs)


def sam_speech_dataset(
        extracted_name='Sam Scholl',
        url='https://drive.google.com/uc?export=download&id=1BTIbI5Rpn4u4Rv19B-njD_S64fYZD8K3',
        url_filename='Sam Scholl.tar.gz',
        speaker=Speaker('Sam Scholl', Gender.MALE),
        **kwargs):
    return _dataset_loader(extracted_name, url, speaker, url_filename, **kwargs)


def susan_speech_dataset(
        extracted_name='Susan Murphy',
        url='https://drive.google.com/uc?export=download&id=1yiDK5pQsLfutZbrxReDuad3XS5RJlYp2',
        url_filename='Susan Murphy.tar.gz',
        speaker=Speaker('Susan Murphy', Gender.FEMALE),
        **kwargs):
    return _dataset_loader(extracted_name, url, speaker, url_filename, **kwargs)


def adrienne_speech_dataset(extracted_name='Adrienne Walker-Heller',
                            url='https://drive.google.com/uc?export=download&id=',
                            url_filename='Adrienne Walker-Heller.tar.gz',
                            speaker=Speaker('Adrienne Walker-Heller', Gender.FEMALE),
                            **kwargs):
    return _dataset_loader(extracted_name, url, speaker, url_filename, **kwargs)


def frank_speech_dataset(extracted_name='Frank Bonacquisti',
                         url='https://drive.google.com/uc?export=download&id=',
                         url_filename='Frank Bonacquisti.tar.gz',
                         speaker=Speaker('Frank Bonacquisti', Gender.FEMALE),
                         **kwargs):
    return _dataset_loader(extracted_name, url, speaker, url_filename, **kwargs)


__all__ = [
    'Speaker', 'Gender', 'lj_speech_dataset', 'm_ailabs_speech_dataset', 'hilary_speech_dataset',
    'heather_speech_dataset', 'beth_speech_dataset', 'beth_custom_speech_dataset',
    'sam_speech_dataset', 'susan_speech_dataset', 'TextSpeechRow',
    'add_predicted_spectrogram_column', 'add_spectrogram_column', 'filter_',
    'normalize_audio_column'
]

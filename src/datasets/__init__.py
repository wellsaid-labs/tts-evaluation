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
        speaker=Speaker('Beth Cameron (Custom)', Gender.FEMALE),
        **kwargs):
    """ Note that this dataset was created from Beth's past VO work that she sent accross.
    """
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
        url='https://drive.google.com/uc?export=download&id=1YHX6yl1kX7lQguxSs4sJ1FPrAS9NZ8O4',
        url_filename='Sean Hannity.tar.gz',
        speaker=Speaker('Sean Hannity', Gender.MALE),
        **kwargs):
    """ Note that this dataset is created and owned by iHeartRadio.
    """
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


def alicia_speech_dataset(
        extracted_name='Alicia Harris',
        url='https://drive.google.com/uc?export=download&id=1x2_XGTTqrwXjSYWRDfGRsoV0aSWDHr6G',
        url_filename='AliciaHarris.tar.gz',
        speaker=Speaker('Alicia Harris', Gender.FEMALE),
        **kwargs):
    return _dataset_loader(extracted_name, url, speaker, url_filename, **kwargs)


def george_speech_dataset(
        extracted_name='George Drake',
        url='https://drive.google.com/uc?export=download&id=1ktZWjaeWoSvz8wckmcF4VGA_Ngrgfq3P',
        url_filename='George Drake.tar.gz',
        speaker=Speaker('George Drake', Gender.MALE),
        **kwargs):
    return _dataset_loader(extracted_name, url, speaker, url_filename, **kwargs)


def megan_speech_dataset(
        extracted_name='Megan Sinclair',
        url='https://drive.google.com/uc?export=download&id=1waUWeXvrgchFjeXMmfBs55obK9u6qr30',
        url_filename='MeganSinclair.tar.gz',
        speaker=Speaker('Megan Sinclair', Gender.FEMALE),
        **kwargs):
    return _dataset_loader(extracted_name, url, speaker, url_filename, **kwargs)


def nadine_speech_dataset(
        extracted_name='Nadine Nagamatsu',
        url='https://drive.google.com/uc?export=download&id=1fwW6oV7x3QYImSfG811vhfjp8jKXVMGZ',
        url_filename='Nadine Nagamatsu.tar.gz',
        speaker=Speaker('Nadine Nagamatsu', Gender.FEMALE),
        **kwargs):
    """ Note that this dataset is compromised due to some audio recording issues.
    """
    return _dataset_loader(extracted_name, url, speaker, url_filename, **kwargs)


def elise_speech_dataset(
        extracted_name='Elise Randall',
        url='https://drive.google.com/uc?export=download&id=1-lbK0J2a9pr-G0NpyxZjcl8Jlz0lvgsc',
        url_filename='EliseRandall.tar.gz',
        speaker=Speaker('Elise Randall', Gender.FEMALE),
        **kwargs):
    return _dataset_loader(extracted_name, url, speaker, url_filename, **kwargs)


def hanuman_speech_dataset(
        extracted_name='Hanuman Welch',
        url='https://drive.google.com/uc?export=download&id=1dU4USVsAd_0aZmjOVCvwmK2_mdQFratZ',
        url_filename='HanumanWelch.tar.gz',
        speaker=Speaker('Hanuman Welch', Gender.MALE),
        **kwargs):
    return _dataset_loader(extracted_name, url, speaker, url_filename, **kwargs)


def jack_speech_dataset(
        extracted_name='Jack Rutkowski',
        url='https://drive.google.com/uc?export=download&id=1n5DhLuvK56Ge57R7maD7Rs4dXVBTBy3l',
        url_filename='JackRutkowski.tar.gz',
        speaker=Speaker('Jack Rutkowski', Gender.MALE),
        **kwargs):
    return _dataset_loader(extracted_name, url, speaker, url_filename, **kwargs)


def mark_speech_dataset(
        extracted_name='Mark Atherlay',
        url='https://drive.google.com/uc?export=download&id=1qi2nRASZXQlzwsfykoaWXtmR_MYFISC5',
        url_filename='Mark Atherlay.tar.gz',
        speaker=Speaker('Mark Atherlay', Gender.MALE),
        **kwargs):
    """ Note that this dataset is only 5.5 hours.
    """
    return _dataset_loader(extracted_name, url, speaker, url_filename, **kwargs)


def steven_speech_dataset(
        extracted_name='Steven Wahlberg',
        url='https://drive.google.com/uc?export=download&id=1osZFUK7_fcnw5zTrSVhGCb5WBZfnGYdT',
        url_filename='StevenWahlberg.tar.gz',
        speaker=Speaker('Steven Wahlberg', Gender.MALE),
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

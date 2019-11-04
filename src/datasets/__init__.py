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
from src.datasets.lj_speech import LINDA_JOHNSON
from src.datasets.lj_speech import lj_speech_dataset
from src.datasets.m_ailabs import ELIZABETH_KLETT
from src.datasets.m_ailabs import ELLIOT_MILLER
from src.datasets.m_ailabs import JUDY_BIEBER
from src.datasets.m_ailabs import m_ailabs_en_uk_speech_dataset
from src.datasets.m_ailabs import m_ailabs_en_us_speech_dataset
from src.datasets.m_ailabs import MARY_ANN
from src.datasets.utils import _dataset_loader
from src.datasets.utils import add_predicted_spectrogram_column
from src.datasets.utils import add_spectrogram_column
from src.datasets.utils import filter_
from src.datasets.utils import normalize_audio_column

# TODO: Consider not using the actors realnames in the codebase in an effort to protect their
# privacy.
# TODO: Consider not using public Google Drive links for safety reasons.

HILARY_NORIEGA = Speaker('Hilary Noriega', Gender.FEMALE)


def hilary_speech_dataset(
        extracted_name='Hilary Noriega',
        url='https://drive.google.com/uc?export=download&id=1VKefPVjDCfc1Qwb-gRHoGh0kyX__uOG8',
        url_filename='Hilary Noriega.tar.gz',
        speaker=HILARY_NORIEGA,
        **kwargs):
    return _dataset_loader(extracted_name, url, speaker, url_filename, **kwargs)


HEATHER_DOE = Speaker('Heather Doe', Gender.FEMALE)


def heather_speech_dataset(
        extracted_name='Heather Doe',
        url='https://drive.google.com/uc?export=download&id=1kqKGkyQq0lA32Rgos0WI9m-widz8g1HY',
        url_filename='Heather Doe.tar.gz',
        speaker=HEATHER_DOE,
        **kwargs):
    return _dataset_loader(extracted_name, url, speaker, url_filename, **kwargs)


BETH_CAMERON_CUSTOM = Speaker('Beth Cameron (Custom)', Gender.FEMALE)


def beth_custom_speech_dataset(
        extracted_name='Beth Cameron (Custom)',
        url='https://drive.google.com/uc?export=download&id=1OJBAtSoaDzdlW9NWUR20F6HJ6U_BXBK2',
        url_filename='Beth Cameron (Custom).tar.gz',
        speaker=BETH_CAMERON_CUSTOM,
        **kwargs):
    return _dataset_loader(extracted_name, url, speaker, url_filename, **kwargs)


BETH_CAMERON = Speaker('Beth Cameron', Gender.FEMALE)


def beth_speech_dataset(
        extracted_name='Beth Cameron',
        url='https://drive.google.com/uc?export=download&id=1A-at3ZI1Aknbr5fVqlDM-rOl3A1It27W',
        url_filename='Beth Cameron.tar.gz',
        speaker=BETH_CAMERON,
        **kwargs):
    return _dataset_loader(extracted_name, url, speaker, url_filename, **kwargs)


SEAN_HANNITY = Speaker('Sean Hannity', Gender.MALE)


def sean_speech_dataset(
        extracted_name='Sean Hannity',
        url='https://drive.google.com/uc?export=download&id=1YHX6yl1kX7lQguxSs4sJ1FPrAS9NZ8O4',
        url_filename='Sean Hannity.tar.gz',
        speaker=SEAN_HANNITY,
        **kwargs):
    return _dataset_loader(extracted_name, url, speaker, url_filename, **kwargs)


SAM_SCHOLL = Speaker('Sam Scholl', Gender.MALE)


def sam_speech_dataset(
        extracted_name='Sam Scholl',
        url='https://drive.google.com/uc?export=download&id=1AvAwYWgUC300l9VNUMeW1Kk0jUHGJxky',
        url_filename='Sam Scholl.tar.gz',
        speaker=SAM_SCHOLL,
        **kwargs):
    return _dataset_loader(extracted_name, url, speaker, url_filename, **kwargs)


SUSAN_MURPHY = Speaker('Susan Murphy', Gender.FEMALE)


def susan_speech_dataset(
        extracted_name='Susan Murphy',
        url='https://drive.google.com/uc?export=download&id=1oHCa6cKcYLQQcmER65ASzSTPFPzsg3JQ',
        url_filename='Susan Murphy.tar.gz',
        speaker=SUSAN_MURPHY,
        **kwargs):
    return _dataset_loader(extracted_name, url, speaker, url_filename, **kwargs)


ADRIENNE_WALKER_HELLER = Speaker('Adrienne Walker-Heller', Gender.FEMALE)


def adrienne_speech_dataset(
        extracted_name='Adrienne Walker-Heller',
        url='https://drive.google.com/uc?export=download&id=1MAypaxctTPlQw5zmYD02uId3ruuGenoW',
        url_filename='Adrienne Walker-Heller.tar.gz',
        speaker=ADRIENNE_WALKER_HELLER,
        **kwargs):
    return _dataset_loader(extracted_name, url, speaker, url_filename, **kwargs)


FRANK_BONACQUISTI = Speaker('Frank Bonacquisti', Gender.MALE)


def frank_speech_dataset(
        extracted_name='Frank Bonacquisti',
        url='https://drive.google.com/uc?export=download&id=1IJLADnQm6Cw8tLJNNqfmDefPj-aVjH9l',
        url_filename='Frank Bonacquisti.tar.gz',
        speaker=FRANK_BONACQUISTI,
        **kwargs):
    return _dataset_loader(extracted_name, url, speaker, url_filename, **kwargs)


ALICIA_HARRIS = Speaker('Alicia Harris', Gender.FEMALE)


def alicia_speech_dataset(
        extracted_name='AliciaHarris',
        url='https://drive.google.com/uc?export=download&id=1x2_XGTTqrwXjSYWRDfGRsoV0aSWDHr6G',
        url_filename='AliciaHarris.tar.gz',
        speaker=ALICIA_HARRIS,
        create_root=True,
        **kwargs):
    return _dataset_loader(extracted_name, url, speaker, url_filename, create_root, **kwargs)


GEORGE_DRAKE = Speaker('George Drake', Gender.MALE)


def george_speech_dataset(
        extracted_name='George Drake, Jr. ',
        url='https://drive.google.com/uc?export=download&id=1ktZWjaeWoSvz8wckmcF4VGA_Ngrgfq3P',
        url_filename='George Drake.tar.gz',
        speaker=GEORGE_DRAKE,
        **kwargs):
    return _dataset_loader(extracted_name, url, speaker, url_filename, **kwargs)


MEGAN_SINCLAIR = Speaker('Megan Sinclair', Gender.FEMALE)


def megan_speech_dataset(
        extracted_name='MeganSinclair',
        url='https://drive.google.com/uc?export=download&id=1waUWeXvrgchFjeXMmfBs55obK9u6qr30',
        url_filename='MeganSinclair.tar.gz',
        speaker=MEGAN_SINCLAIR,
        create_root=True,
        **kwargs):
    return _dataset_loader(extracted_name, url, speaker, url_filename, create_root, **kwargs)


NADINE_NAGAMATSU = Speaker('Nadine Nagamatsu', Gender.FEMALE)


def nadine_speech_dataset(
        extracted_name='Nadine Nagamatsu',
        url='https://drive.google.com/uc?export=download&id=1fwW6oV7x3QYImSfG811vhfjp8jKXVMGZ',
        url_filename='Nadine Nagamatsu.tar.gz',
        speaker=NADINE_NAGAMATSU,
        **kwargs):
    return _dataset_loader(extracted_name, url, speaker, url_filename, **kwargs)


ELISE_RANDALL = Speaker('Elise Randall', Gender.FEMALE)


def elise_speech_dataset(
        extracted_name='EliseRandall',
        url='https://drive.google.com/uc?export=download&id=1-lbK0J2a9pr-G0NpyxZjcl8Jlz0lvgsc',
        url_filename='EliseRandall.tar.gz',
        speaker=ELISE_RANDALL,
        create_root=True,
        **kwargs):
    return _dataset_loader(extracted_name, url, speaker, url_filename, create_root, **kwargs)


HANUMAN_WELCH = Speaker('Hanuman Welch', Gender.MALE)


def hanuman_speech_dataset(
        extracted_name='Hanuman Welch',
        url='https://drive.google.com/uc?export=download&id=1dU4USVsAd_0aZmjOVCvwmK2_mdQFratZ',
        url_filename='HanumanWelch.tar.gz',
        speaker=HANUMAN_WELCH,
        create_root=True,
        **kwargs):
    return _dataset_loader(extracted_name, url, speaker, url_filename, create_root, **kwargs)


JACK_RUTKOWSKI = Speaker('Jack Rutkowski', Gender.MALE)


def jack_speech_dataset(
        extracted_name='JackRutkowski',
        url='https://drive.google.com/uc?export=download&id=1n5DhLuvK56Ge57R7maD7Rs4dXVBTBy3l',
        url_filename='JackRutkowski.tar.gz',
        speaker=JACK_RUTKOWSKI,
        create_root=True,
        **kwargs):
    return _dataset_loader(extracted_name, url, speaker, url_filename, create_root, **kwargs)


MARK_ATHERLAY = Speaker('Mark Atherlay', Gender.MALE)


def mark_speech_dataset(
        extracted_name='MarkAtherlay',
        url='https://drive.google.com/uc?export=download&id=1qi2nRASZXQlzwsfykoaWXtmR_MYFISC5',
        url_filename='Mark Atherlay.tar.gz',
        speaker=MARK_ATHERLAY,
        create_root=True,
        **kwargs):
    return _dataset_loader(extracted_name, url, speaker, url_filename, create_root, **kwargs)


STEVEN_WAHLBERG = Speaker('Steven Wahlberg', Gender.MALE)


def steven_speech_dataset(
        extracted_name='StevenWahlberg',
        url='https://drive.google.com/uc?export=download&id=1osZFUK7_fcnw5zTrSVhGCb5WBZfnGYdT',
        url_filename='StevenWahlberg.tar.gz',
        speaker=STEVEN_WAHLBERG,
        create_root=True,
        **kwargs):
    return _dataset_loader(extracted_name, url, speaker, url_filename, create_root, **kwargs)


__all__ = [
    'Speaker', 'Gender', 'lj_speech_dataset', 'm_ailabs_en_us_speech_dataset',
    'm_ailabs_en_uk_speech_dataset', 'hilary_speech_dataset', 'heather_speech_dataset',
    'beth_speech_dataset', 'beth_custom_speech_dataset', 'sam_speech_dataset',
    'susan_speech_dataset', 'sean_speech_dataset', 'TextSpeechRow',
    'add_predicted_spectrogram_column', 'add_spectrogram_column', 'filter_',
    'normalize_audio_column', 'STEVEN_WAHLBERG', 'MARK_ATHERLAY', 'JACK_RUTKOWSKI', 'HANUMAN_WELCH',
    'ELISE_RANDALL', 'NADINE_NAGAMATSU', 'MEGAN_SINCLAIR', 'GEORGE_DRAKE', 'ALICIA_HARRIS',
    'FRANK_BONACQUISTI', 'ADRIENNE_WALKER_HELLER', 'SUSAN_MURPHY', 'SAM_SCHOLL', 'SEAN_HANNITY',
    'BETH_CAMERON', 'BETH_CAMERON_CUSTOM', 'HEATHER_DOE', 'HILARY_NORIEGA', 'MARY_ANN',
    'JUDY_BIEBER', 'ELLIOT_MILLER', 'ELIZABETH_KLETT', 'LINDA_JOHNSON'
]

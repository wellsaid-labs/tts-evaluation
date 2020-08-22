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

__all__ = [
    'Speaker',
    'Gender',
    'lj_speech_dataset',
    'm_ailabs_en_us_speech_dataset',
    'm_ailabs_en_uk_speech_dataset',
    'TextSpeechRow',
    'add_predicted_spectrogram_column',
    'add_spectrogram_column',
    'filter_',
    'normalize_audio_column',
]

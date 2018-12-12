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
from src.datasets.lj_speech import lj_speech_dataset
from src.datasets.m_ailabs import m_ailabs_speech_dataset
from src.datasets.hilary import hilary_dataset
from src.datasets.constants import Speaker
from src.datasets.constants import Gender
from src.datasets.constants import TextSpeechRow
from src.datasets.constants import SpectrogramTextSpeechRow
from src.datasets.process import compute_spectrograms

__all__ = [
    'Speaker', 'Gender', 'lj_speech_dataset', 'm_ailabs_speech_dataset', 'hilary_dataset',
    'TextSpeechRow', 'SpectrogramTextSpeechRow', 'compute_spectrograms'
]

from src.datasets.lj_speech import lj_speech_dataset
from src.datasets.m_ailabs import m_ailabs_speech_dataset
from src.datasets.hillary import hillary_dataset

# TODO: Support more datasets
# * English Bible Speech Dataset
#   https://www.kaggle.com/bryanpark/the-world-english-bible-speech-dataset
# *	CMU Arctic Speech Synthesis Dataset
#   http://festvox.org/cmu_arctic/
# * Synthetic Wave Dataset with SoX e.g. [sine, square, triangle, sawtooth, trapezium, exp, brow]
# * VCTK Dataset
# * Voice Conversion Challenge (VCC) 2016 dataset
# * Blizzard Dataset
# * JSUT dataset

__all__ = ['lj_speech_dataset', 'm_ailabs_speech_dataset', 'hillary_dataset']

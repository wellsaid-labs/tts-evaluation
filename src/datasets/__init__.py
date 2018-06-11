from src.datasets.lj_speech import lj_speech_dataset

# TODO: Support more datasets
# * English Bible Speech Dataset
#   https://www.kaggle.com/bryanpark/the-world-english-bible-speech-dataset
# *	CMU Arctic Speech Synthesis Dataset
#   http://festvox.org/cmu_arctic/
# * Sine Wave Dataset
#   def sine_wave(freq, length, sample_rate=sample_rate):
#       return np.sin(np.arange(length) * 2 * math.pi * freq / sample_rate).astype(np.float32)
# * VCTK Dataset
# * Voice Conversion Challenge (VCC) 2016 dataset
# * Blizzard Dataset
# * JSUT dataset

__all__ = ['lj_speech_dataset']

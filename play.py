from functools import partial

import numpy as np
import tensorflow as tf
import librosa

from tensorflow.contrib.signal.python.ops import window_ops

tf.enable_eager_execution()

# Conversions
_mel_basis = None
_inv_mel_basis = None


def _linear_to_mel(spectogram, **kwargs):
    global _mel_basis
    if _mel_basis is None:
        _mel_basis = _build_mel_basis(**kwargs)
    return np.dot(_mel_basis, spectogram)


def _mel_to_linear(mel_spectrogram, **kwargs):
    global _inv_mel_basis
    if _inv_mel_basis is None:
        _inv_mel_basis = np.linalg.pinv(_build_mel_basis(**kwargs))
    return np.maximum(1e-10, np.dot(_inv_mel_basis, mel_spectrogram))


def _build_mel_basis(sample_rate, fft_size, num_mels):
    return librosa.filters.mel(
        sample_rate,
        fft_size,
        n_mels=num_mels,
        fmin=125,
        fmax=7600,
    )


def _read_audio(filename, sample_rate=None):
    audio, sample_rate = librosa.core.load(filename, sr=sample_rate, mono=True)
    audio = np.expand_dims(audio, axis=1)
    return audio, sample_rate


signals, sample_rate = _read_audio('tests/_test_data/lj_speech.wav')
signals = tf.convert_to_tensor(signals)

# [signal_length, batch_size] -> [batch_size, signal_length]
signals = tf.transpose(signals)

spectrograms = tf.contrib.signal.stft(
    signals,
    frame_length=1200,
    frame_step=300,
    window_fn=partial(window_ops.hann_window, periodic=True),
)
magnitude_spectrograms = tf.abs(spectrograms)
magnitude_spectrograms = magnitude_spectrograms.numpy()
num_spectrogram_bins = magnitude_spectrograms.shape[-1]
fft_size = (num_spectrogram_bins - 1) * 2
print(magnitude_spectrograms.shape)
magnitude_spectrograms = np.transpose(magnitude_spectrograms, (0, 2, 1))
print('magnitude_spectrograms', magnitude_spectrograms.shape)
mel = _linear_to_mel(
    magnitude_spectrograms, sample_rate=sample_rate, fft_size=fft_size, num_mels=80)
print(mel.shape)
mel = np.transpose(mel, (1, 0, 2))
linear = _mel_to_linear(mel, sample_rate=sample_rate, fft_size=fft_size, num_mels=80)
print(linear.sum())
print(magnitude_spectrograms.sum())
print(magnitude_spectrograms.shape)
print(linear.shape)

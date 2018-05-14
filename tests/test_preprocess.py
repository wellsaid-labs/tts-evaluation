import os

import numpy as np
import tensorflow as tf

from tensorflow.contrib.framework.python.ops import audio_ops

from src.preprocess import read_audio
from src.preprocess import wav_to_log_mel_spectrogram
from src.preprocess import log_mel_spectrogram_to_wav
from src.preprocess import _milliseconds_to_samples
from src.preprocess import mu_law_quantize


def test_mulaw():
    assert mu_law_quantize(-1.0, 2) == 0
    assert mu_law_quantize(-0.5, 2) == 0
    assert mu_law_quantize(-0.001, 2) == 0
    assert mu_law_quantize(0.0, 2) == 1
    assert mu_law_quantize(0.0001, 2) == 1
    assert mu_law_quantize(0.5, 2) == 1
    assert mu_law_quantize(0.99999, 2) == 1
    assert mu_law_quantize(1.0, 2) == 2


def test_librosa_tf_decode_wav():
    """ Librosa provides a more flexible API for decoding WAVs. To ensure consistency with TF, we
    test the output is the same.
    """
    wav_filename = 'tests/_test_data/lj_speech.wav'

    audio_binary = tf.read_file(wav_filename)
    tf_audio, _ = audio_ops.decode_wav(audio_binary)

    audio, _ = read_audio(wav_filename, sample_rate=None)

    np.testing.assert_array_equal(tf_audio[:, 0], audio)


def test_wav_to_log_mel_spectrogram_smoke():
    """ Smoke test to ensure everything runs.
    """
    frame_size = 50
    frame_hop = 12.5
    num_mel_bins = 80
    wav_filename = 'tests/_test_data/lj_speech.wav'
    signal, sample_rate = read_audio(wav_filename)
    log_mel_spectrogram, right_pad = wav_to_log_mel_spectrogram(
        signal, sample_rate, frame_size=frame_size, frame_hop=frame_hop, num_mel_bins=num_mel_bins)
    frame_hop = _milliseconds_to_samples(frame_hop, sample_rate)

    assert log_mel_spectrogram.shape == (607, num_mel_bins)
    assert signal.shape == (182015,)
    assert int(signal.shape[0] + right_pad) / int(log_mel_spectrogram.shape[0]) == frame_hop


def test_log_mel_spectrogram_to_wav_smoke():
    """ Smoke test to ensure everything runs.
    """
    wav_filename = 'tests/_test_data/lj_speech.wav'
    new_wav_filename = 'tests/_test_data/lj_speech_reconstructed.wav'
    signal, sample_rate = read_audio(wav_filename)
    log_mel_spectrogram, _ = wav_to_log_mel_spectrogram(signal, sample_rate)
    log_mel_spectrogram_to_wav(log_mel_spectrogram, new_wav_filename, sample_rate, log=True)

    assert os.path.isfile(new_wav_filename)

    # Clean up
    os.remove(new_wav_filename)

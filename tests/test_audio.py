import os

import numpy as np
import tensorflow as tf

from tensorflow.contrib.framework.python.ops import audio_ops

from src.audio import read_audio
from src.audio import wav_to_log_mel_spectrogram
from src.audio import log_mel_spectrogram_to_wav
from src.audio import mu_law_quantize
from src.audio import mu_law
from src.audio import inverse_mu_law
from src.audio import inverse_mu_law_quantize


def test_mu_law_quantize_edge_cases():
    assert mu_law_quantize(-1.0, 2) == 0
    assert mu_law_quantize(-0.5, 2) == 0
    assert mu_law_quantize(-0.001, 2) == 0
    assert mu_law_quantize(0.0, 2) == 1
    assert mu_law_quantize(0.0001, 2) == 1
    assert mu_law_quantize(0.5, 2) == 1
    assert mu_law_quantize(0.99999, 2) == 1
    assert mu_law_quantize(1.0, 2) == 2


def test_mu_law_forward_backward_quantize():
    # forward/backward correctness for quantize
    for mu in [128, 256, 512]:
        for x, y in [(-1.0, 0), (0.0, mu // 2), (0.99999, mu - 1)]:
            y_hat = mu_law_quantize(x, mu)
            err = np.abs(x - inverse_mu_law_quantize(y_hat, mu))
            assert np.allclose(y, y_hat)
            # have small quantize error
            assert err <= 0.1


def test_mu_law_torch():
    import torch
    torch.manual_seed(1234)
    for mu in [128, 256, 512]:
        x = torch.rand(10)
        y = mu_law(x, mu)
        x_hat = inverse_mu_law(y, mu)
        assert np.allclose(x, x_hat)
        inverse_mu_law_quantize(mu_law_quantize(x))


def test_mu_law_nd_array():
    # ndarray input
    for mu in [128, 256, 512]:
        x = np.random.rand(10)
        y = mu_law(x, mu)
        x_hat = inverse_mu_law(y, mu)
        assert np.allclose(x, x_hat)
        inverse_mu_law_quantize(mu_law_quantize(x))


def test_mu_law_forward_backward():
    np.random.seed(1234)
    # forward/backward correctness
    for mu in [128, 256, 512]:
        for x in np.random.rand(100):
            y = mu_law(x, mu)
            assert y >= 0 and y <= 1
            x_hat = inverse_mu_law(y, mu)
            assert np.allclose(x, x_hat)


def test_librosa_tf_decode_wav():
    """ Librosa provides a more flexible API for decoding WAVs. To ensure consistency with TF, we
    test the output is the same.
    """
    wav_filename = 'tests/_test_data/lj_speech.wav'

    audio_binary = tf.read_file(wav_filename)
    tf_audio, _ = audio_ops.decode_wav(audio_binary)

    audio = read_audio(wav_filename, sample_rate=None)

    np.testing.assert_array_equal(tf_audio[:, 0], audio)


def test_wav_to_log_mel_spectrogram_smoke():
    """ Smoke test to ensure everything runs.
    """
    frame_size = 1200
    frame_hop = 300
    num_mel_bins = 80
    wav_filename = 'tests/_test_data/lj_speech.wav'
    sample_rate = 22050
    signal = read_audio(wav_filename, sample_rate)
    log_mel_spectrogram, right_pad = wav_to_log_mel_spectrogram(
        signal, sample_rate, frame_size=frame_size, frame_hop=frame_hop, num_mel_bins=num_mel_bins)

    assert log_mel_spectrogram.shape[1] == num_mel_bins
    assert len(log_mel_spectrogram.shape) == 2
    assert len(signal.shape) == 1
    assert int(signal.shape[0] + right_pad) / int(log_mel_spectrogram.shape[0]) == frame_hop


def test_log_mel_spectrogram_to_wav_smoke():
    """ Smoke test to ensure everything runs.
    """
    wav_filename = 'tests/_test_data/lj_speech.wav'
    new_wav_filename = 'tests/_test_data/lj_speech_reconstructed.wav'
    sample_rate = 22050
    signal = read_audio(wav_filename, sample_rate)
    log_mel_spectrogram, _ = wav_to_log_mel_spectrogram(signal, sample_rate)
    log_mel_spectrogram_to_wav(log_mel_spectrogram, new_wav_filename, sample_rate, log=True)

    assert os.path.isfile(new_wav_filename)

    # Clean up
    os.remove(new_wav_filename)

import os

import numpy as np
import pytest
import torch

from src.audio import read_audio
from src.audio import get_log_mel_spectrogram
from src.audio import griffin_lim
from src.audio import mu_law_encode
from src.audio import mu_law_decode


def test_mu_law_encode_edge_cases():
    # INSPIRED BY:
    # https://github.com/r9y9/nnmnkwii/blob/master/tests/test_preprocessing.py
    BINS = 3

    # NOTE: The ranges are not exact due to the non-linearity of mu-law
    # ~[-1.0, -0.4] => 0
    assert mu_law_encode(-1.0, BINS) == 0
    assert mu_law_encode(-0.4, BINS) == 0

    # ~(-0.3, 0.3) => 1
    assert mu_law_encode(-0.3, BINS) == 1
    assert mu_law_encode(-0.001, BINS) == 1
    assert mu_law_encode(0.0, BINS) == 1
    assert mu_law_encode(0.0001, BINS) == 1
    assert mu_law_encode(0.3, BINS) == 1

    # ~[0.4, 1.0] => 1
    assert mu_law_encode(0.4, BINS) == 2
    assert mu_law_encode(0.99999, BINS) == 2
    assert mu_law_encode(1.0, BINS) == 2

    assert mu_law_decode(2, BINS) == 1
    assert mu_law_decode(1, BINS) == 0
    assert mu_law_decode(0, BINS) == -1


def test_mu_law_forward_backward_quantize():
    # INSPIRED BY:
    # https://github.com/r9y9/nnmnkwii/blob/master/tests/test_preprocessing.py
    # forward/backward correctness for quantize
    for bins in [128, 256, 512]:
        for x, y in [(-1.0, 0), (0.0, bins // 2), (0.99999, bins - 1)]:
            y_hat = mu_law_encode(x, bins)
            err = np.abs(x - mu_law_decode(y_hat, bins))
            assert np.allclose(y, y_hat, rtol=1e-1, atol=0.05)
            # have small quantize error
            assert err <= 0.1


def test_mu_law_torch():
    # INSPIRED BY:
    # https://github.com/r9y9/nnmnkwii/blob/master/tests/test_preprocessing.py
    torch.manual_seed(1234)
    for bins in [128, 256, 512]:
        x = torch.rand(10)
        y = mu_law_encode(x, bins)
        x_hat = mu_law_decode(y, bins)
        assert np.allclose(x, x_hat, rtol=1e-1, atol=0.05)


def test_mu_law_torch_short_tensor():
    BINS = 256

    torch.manual_seed(1234)
    x = torch.ShortTensor([1, 2, 3])
    y = mu_law_encode(x, BINS)
    x_hat = mu_law_decode(y, BINS)
    assert np.allclose(x, x_hat, rtol=1e-1, atol=0.05)


def test_mu_law_nd_array():
    # INSPIRED BY:
    # https://github.com/r9y9/nnmnkwii/blob/master/tests/test_preprocessing.py
    # ndarray input
    for bins in [128, 256, 512]:
        x = np.random.rand(10)
        y = mu_law_encode(x, bins)
        x_hat = mu_law_decode(y, bins)
        assert np.allclose(x, x_hat, rtol=1e-1, atol=0.05)


def test_mu_law_forward_backward_random():
    # INSPIRED BY:
    # https://github.com/r9y9/nnmnkwii/blob/master/tests/test_preprocessing.py
    np.random.seed(1234)
    # forward/backward correctness
    for bins in [128, 256, 512]:
        for x in np.random.rand(100):
            y = mu_law_encode(x, bins)
            assert y >= 0 and y <= bins
            x_hat = mu_law_decode(y, bins)
            assert np.allclose(x, x_hat, rtol=1e-1, atol=0.05)


def test_mu_law_forward_backward():
    # INSPIRED BY:
    # https://github.com/ibab/tensorflow-wavenet/blob/master/test/test_mu_law.py
    BINS = 256

    # generate every possible quantized level.
    x = np.array(range(BINS), dtype=np.int)

    # Decode into floating-point scalar.
    decoded = mu_law_decode(x, BINS)
    # Encode back into an integer quantization level.
    encoded = mu_law_encode(decoded, BINS)

    assert np.allclose(x, encoded)


def test_mu_law_min_max_range():
    # INSPIRED BY:
    # https://github.com/ibab/tensorflow-wavenet/blob/master/test/test_mu_law.py
    BINS = 256

    # Generate every possible quantized level.
    x = np.array(range(BINS), dtype=np.int)

    # Decode into floating-point scalar.
    decoded = mu_law_decode(x, BINS)

    # Our range should be exactly [-1,1].
    max_val = np.max(decoded)
    min_val = np.min(decoded)
    EPSILON = 1e-10
    assert max_val == pytest.approx(1.0, abs=EPSILON)
    assert min_val == pytest.approx(-1.0, abs=EPSILON)


def test_mu_law_shift():
    # INSPIRED BY:
    # https://github.com/ibab/tensorflow-wavenet/blob/master/test/test_mu_law.py
    BINS = 256
    x = np.linspace(-1, 1, 1000).astype(np.float32)
    encoded = mu_law_encode(x, BINS)
    decoded = mu_law_decode(encoded, BINS)

    # Detect non-unity scaling and non-zero shift in the roundtripped
    # signal by asserting that slope = 1 and y-intercept = 0 of line fit to
    # roundtripped vs x values.
    coeffs = np.polyfit(x, decoded, 1)
    slope = coeffs[0]
    y_intercept = coeffs[1]
    EPSILON = 1e-4
    assert slope == pytest.approx(1.0, abs=EPSILON)
    assert y_intercept == pytest.approx(0.0, abs=EPSILON)


def test_mu_law_forward_backward_linspace():
    # INSPIRED BY:
    # https://github.com/ibab/tensorflow-wavenet/blob/master/test/test_mu_law.py
    x = np.linspace(-1, 1, 1000).astype(np.float32)
    BINS = 256

    # Test whether decoded signal is roughly equal to
    # what was encoded before
    encoded = mu_law_encode(x, BINS)
    decoded = mu_law_decode(encoded, BINS)

    assert np.allclose(x, decoded, rtol=1e-1, atol=0.05)

    # Make sure that re-encoding leaves the waveform invariant
    encoded = mu_law_encode(x, BINS)
    decoded_other = mu_law_decode(encoded, BINS)

    assert np.allclose(decoded, decoded_other)


def test_mu_law_is_surjective():
    # INSPIRED BY:
    # https://github.com/ibab/tensorflow-wavenet/blob/master/test/test_mu_law.py
    x = np.linspace(-1, 1, 10000).astype(np.float32)
    BINS = 123
    encoded = mu_law_encode(x, BINS)
    assert len(np.unique(encoded)) == BINS


def test_mu_law_precomputed():
    # INSPIRED BY:
    # https://github.com/ibab/tensorflow-wavenet/blob/master/test/test_mu_law.py
    BINS = 256
    x = np.array([-1.0, 1.0, 0.6, -0.25, 0.01, 0.33, -0.9999, 0.42, 0.1, -0.45]).astype(np.float32)
    encoded_manual = np.array([0, 255, 243, 32, 157, 230, 0, 235, 203, 18]).astype(np.int32)

    encoded = mu_law_encode(x, BINS)

    assert np.array_equal(encoded_manual, encoded)


def test_log_mel_spectrogram_smoke():
    """ Smoke test to ensure everything runs.
    """
    frame_size = 1200
    frame_hop = 300
    wav_filename = 'tests/_test_data/lj_speech.wav'
    sample_rate = 22050
    signal = read_audio(wav_filename, sample_rate)
    log_mel_spectrogram, padding = get_log_mel_spectrogram(
        signal, sample_rate, frame_size=frame_size, frame_hop=frame_hop)

    assert log_mel_spectrogram.dtype == np.float32
    assert len(log_mel_spectrogram.shape) == 2
    assert len(signal.shape) == 1
    assert int(signal.shape[0] + sum(padding)) / int(log_mel_spectrogram.shape[0]) == frame_hop


def test_griffin_lim_smoke():
    """ Smoke test to ensure everything runs.
    """
    wav_filename = 'tests/_test_data/lj_speech.wav'
    new_wav_filename = 'tests/_test_data/lj_speech_reconstructed.wav'
    sample_rate = 22050
    signal = read_audio(wav_filename, sample_rate)
    log_mel_spectrogram, _ = get_log_mel_spectrogram(signal, sample_rate)
    griffin_lim(log_mel_spectrogram, new_wav_filename, sample_rate)

    assert os.path.isfile(new_wav_filename)

    # Clean up
    os.remove(new_wav_filename)

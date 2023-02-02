import math

import config as cf
import librosa
import numpy as np
import pytest
import torch
import torchnlp.random
from torchnlp.encoders.text import SequenceBatch

import lib
import run
from run._config.data import _get_loudness_annotation
from run.data._loader import Alignment
from run.train.spectrogram_model import _data
from tests._utils import assert_almost_equal


@pytest.fixture(autouse=True, scope="module")
def run_around_tests():
    """Set a basic configuration."""
    run._config.configure()
    yield
    cf.purge()


def test__random_nonoverlapping_alignments():
    """Test `_data._random_nonoverlapping_alignments` on a basic case."""
    with torchnlp.random.fork_rng(123):
        make = lambda a, b: Alignment((a, b), (a, b), (a, b))
        alignments = Alignment.stow([make(0, 1), make(1, 2), make(2, 3), make(3, 4), make(4, 5)])
        intervals = _data._random_nonoverlapping_alignments(alignments, 2, 0)
        assert intervals == (
            Alignment(script=(0, 4), audio=(0.0, 4.0), transcript=(0, 4)),
            Alignment(script=(4, 5), audio=(4.0, 5.0), transcript=(4, 5)),
        )


def test__random_nonoverlapping_alignments__no_intervals():
    """Test `_data._random_nonoverlapping_alignments` returns no intervals consistently if
    `min_no_intervals_prob` is 100%."""
    make = lambda a, b: Alignment((a, b), (a, b), (a, b))
    alignments = Alignment.stow([make(0, 1), make(1, 2), make(2, 3), make(3, 4), make(4, 5)])
    for _ in range(10):
        intervals = _data._random_nonoverlapping_alignments(alignments, 3, 1)
        assert len(intervals) == 0


def test__random_nonoverlapping_alignments__overlapping():
    """Test `_data._random_nonoverlapping_alignments` returns no intervals consistently if all
    the alignments are overlapping."""
    make = lambda a, b: Alignment((a, b), (a, b), (a, b))
    alignments = Alignment.stow([make(0, 0), make(0, 0), make(0, 0), make(0, 0), make(0, 0)])
    for _ in range(10):
        intervals = _data._random_nonoverlapping_alignments(alignments, 3, 0)
        assert len(intervals) == 0


def test__get_loudness():
    """Test `_data._get_loudness` slices, measures, and rounds loudness correctly."""
    sample_rate = 1000
    length = 10
    implementation = "K-weighting"
    precision = 5
    block_size = 0.4
    meter = lib.audio.get_pyloudnorm_meter(
        sample_rate=sample_rate, filter_class=implementation, block_size=block_size
    )
    with torchnlp.random.fork_rng(1234):
        audio = np.random.rand(sample_rate * length) * 2 - 1
        alignment = Alignment((0, length), (0, length), (0, length))
        loudness = _data._get_loudness_annotation(
            audio=audio,
            alignment=alignment,
            block_size=block_size,
            precision=precision,
            sample_rate=sample_rate,
            filter_class=implementation,
            get_anno=_get_loudness_annotation,
        )
        assert loudness is not None
        assert math.isfinite(loudness)
        assert round(meter.integrated_loudness(audio), precision) == loudness


def test__signals_to_spectrograms():
    """Test `_data._signals_to_spectrograms` is invariant to the batch size."""
    fft_length = 2048
    hop_length = fft_length // 4
    window = torch.from_numpy(librosa.filters.get_window("hann", fft_length)).float()
    batch_size = 16
    signals = []
    for i in range(batch_size):
        signal = lib.audio.pad_remainder(torch.randn(2400 + i * 300).numpy(), hop_length)
        signals.append(torch.from_numpy(signal))
    spectrogram, spectrogram_mask = _data._signals_to_spectrograms(
        signals, fft_length=fft_length, frame_hop=hop_length, window=window
    )
    module = cf.partial(lib.audio.SignalTodBMelSpectrogram)(
        fft_length=fft_length, frame_hop=hop_length, window=window
    )
    for i in range(batch_size):
        result = module(signals[i], aligned=True)
        expected = spectrogram.tensor[:, i][: spectrogram.lengths[:, i]]
        assert_almost_equal(expected, result, decimal=5)
        assert spectrogram_mask.tensor[:, i].sum() == spectrogram.lengths[:, i]
        assert spectrogram_mask.tensor[:, i].sum() == spectrogram_mask.lengths[:, i]


def test__get_normalized_half_gaussian():
    """Test `_data._get_normalized_half_gaussian` generates the left-side of a gaussian
    distribution normalized from 0 to 1."""
    expected_ = [0.0015184, 0.0070632, 0.0292409, 0.0952311, 0.2443498, 0.4946532, 0.7909854, 1.0]
    expected = torch.tensor(expected_)
    assert_almost_equal(_data._get_normalized_half_gaussian(8, 2), expected)


def test__make_stop_token():
    """Test `_data._make_stop_token` makes a batched stop token."""
    spectrogram = SequenceBatch(torch.ones(8, 4, 16), torch.tensor([2, 4, 6, 8]).unsqueeze(0))
    stop_token = _data._make_stop_token(spectrogram, 6, 2)
    assert_almost_equal(stop_token.lengths, spectrogram.lengths)
    expected = [
        [0.7910, 0.2445, 0.0363, 0.0000],
        [1.0000, 0.4947, 0.0966, 0.0000],
        [0.0000, 0.7910, 0.2445, 0.0363],
        [0.0000, 1.0000, 0.4947, 0.0966],
        [0.0000, 0.0000, 0.7910, 0.2445],
        [0.0000, 0.0000, 1.0000, 0.4947],
        [0.0000, 0.0000, 0.0000, 0.7910],
        [0.0000, 0.0000, 0.0000, 1.0000],
    ]
    assert_almost_equal(stop_token.tensor, torch.tensor(expected), decimal=4)

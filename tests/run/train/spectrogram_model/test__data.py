import collections
import math
import typing

import config as cf
import librosa
import numpy as np
import pytest
import torch
import torchnlp.random
from torchnlp.encoders.text import SequenceBatch

import lib
import run
from run.data._loader import Alignment
from run.train.spectrogram_model import _data
from tests._utils import assert_almost_equal, assert_uniform_distribution


@pytest.fixture(autouse=True, scope="module")
def run_around_tests():
    """Set a basic configuration."""
    run._config.configure()
    yield
    cf.purge()


def test__random_nonoverlapping_alignments():
    """Test `_data._random_nonoverlapping_alignments` samples uniformly given a uniform
    distribution of alignments."""
    make = lambda a, b: Alignment((a, b), (a, b), (a, b))
    alignments = Alignment.stow([make(0, 1), make(1, 2), make(2, 3), make(3, 4), make(4, 5)])
    counter: typing.Counter[int] = collections.Counter()
    for i in range(100000):
        for sample in _data._random_nonoverlapping_alignments(alignments, 3):
            for i in range(sample.script[0], sample.script[1]):
                counter[i] += 1
    assert set(counter.keys()) == set(range(0, 5))
    assert_uniform_distribution(counter, abs=0.02)


def test__random_nonoverlapping_alignments__empty():
    """Test `_data._random_nonoverlapping_alignments` handles empty list."""
    input_ = Alignment.stow([])
    assert _data._random_nonoverlapping_alignments(input_, 3) == tuple()


def test__random_nonoverlapping_alignments__large_max():
    """Test `_data._random_nonoverlapping_alignments` handles a large maximum."""
    make = lambda a, b: Alignment((a, b), (a, b), (a, b))
    with torchnlp.random.fork_rng(1234):
        alignments = Alignment.stow(
            [make(0, 1), make(1, 2), make(2, 3), make(3, 4), make(4, 5)],
        )
        assert len(_data._random_nonoverlapping_alignments(alignments, 1000000)) == 6


def test__get_loudness():
    """Test `_data._get_loudnes` slices, measures, and rounds loudness correctly."""
    sample_rate = 1000
    length = 10
    implementation = "K-weighting"
    precision = 5
    block_size = 0.4
    meter = lib.audio.get_pyloudnorm_meter(
        sample_rate=sample_rate, filter_class=implementation, block_size=block_size
    )
    with torchnlp.random.fork_rng(12345):
        audio = np.random.rand(sample_rate * length) * 2 - 1  # type: ignore
        alignment = Alignment((0, length), (0, length), (0, length))
        loundess = _data._get_loudness(
            audio=audio,
            alignment=alignment,
            block_size=block_size,
            precision=precision,
            sample_rate=sample_rate,
            filter_class=implementation,
        )
        assert loundess is not None
        assert math.isfinite(loundess)
        assert round(meter.integrated_loudness(audio), precision) == loundess


def test__get_char_to_word():
    """Test `_data._get_char_to_word` maps characters to words correctly."""
    nlp = lib.text.load_en_english()
    doc = nlp("It was time to present the present abcdefghi.")
    char_to_word = _data._get_char_to_word(doc)
    expected = [0, 0, -1, 1, 1, 1, -1, 2, 2, 2, 2, -1, 3, 3, -1, 4, 4, 4, 4, 4, 4, 4, -1, 5, 5, 5]
    expected += [-1, 6, 6, 6, 6, 6, 6, 6, -1, 7, 7, 7, 7, 7, 7, 7, 7, 7, 8]
    assert char_to_word == expected


def test__get_word_vectors():
    """Test `_data._get_word_vectors` maps word vectors onto characters."""
    text = "It was time to present the present abcdefghi."
    doc = lib.text.load_en_core_web_md()(text)
    char_to_word = _data._get_char_to_word(doc)
    word_vectors = _data._get_word_vectors(char_to_word, doc)

    assert word_vectors.shape == (len(text), 300)
    assert word_vectors[0].sum() != 0

    # NOTE: `-1`/`7` and "present"/"present" have the same word vector.
    expected = (len(set(char_to_word)) - 2, 300)
    assert np.unique(word_vectors, axis=0).shape == expected  # type: ignore

    # NOTE: OOV words and non-word characters should have a zero vectors.
    slice_ = slice(-11, -1)
    assert char_to_word[slice_] == [-1, 7, 7, 7, 7, 7, 7, 7, 7, 7]
    assert word_vectors[slice_].sum() == 0


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

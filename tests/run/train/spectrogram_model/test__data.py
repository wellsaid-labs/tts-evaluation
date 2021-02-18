import collections
import math
import typing

import hparams
import librosa
import numpy as np
import pytest
import torch
import torchnlp.random
from torchnlp.encoders.text import SequenceBatch

import lib
import run
from lib.datasets import Alignment
from run import _spectrogram_model
from run._spectrogram_model import _signals_to_spectrograms
from tests._utils import TEST_DATA_PATH, assert_almost_equal, assert_uniform_distribution


@pytest.fixture(autouse=True)
def run_around_tests():
    """ Set a basic configuration. """
    run._config.configure()
    yield
    hparams.clear_config()


def test_input_encoder():
    """ Test `_spectrogram_model.InputEncoder` handles a basic case. """
    graphemes = ["abc", "def"]
    phonemes = ["ˈ|eɪ|b|ˌ|iː|s|ˈ|iː|", "d|ˈ|ɛ|f"]
    phoneme_separator = "|"
    speakers = [lib.datasets.MARK_ATHERLAY, lib.datasets.MARY_ANN]
    encoder = _spectrogram_model.InputEncoder(graphemes, phonemes, speakers, phoneme_separator)
    input_ = _spectrogram_model.DecodedInput("a", "ˈ|eɪ", lib.datasets.MARK_ATHERLAY)
    assert encoder._get_case("A") == encoder._CASE_LABELS[0]
    assert encoder._get_case("a") == encoder._CASE_LABELS[1]
    assert encoder._get_case("1") == encoder._CASE_LABELS[2]
    encoded = encoder.encode(input_)
    assert torch.equal(encoded[0], torch.tensor([5]))
    assert torch.equal(encoded[1], torch.tensor([1]))
    assert torch.equal(encoded[2], torch.tensor([5, 6]))
    assert torch.equal(encoded[3], torch.tensor([0]))
    assert encoder.decode(encoded) == input_


def test__random_nonoverlapping_alignments():
    """Test `_spectrogram_model._random_nonoverlapping_alignments` samples uniformly given a uniform
    distribution of alignments."""
    make = lambda a, b: Alignment((a, b), (a, b), (a, b))
    alignments = tuple([make(0, 1), make(1, 2), make(2, 3), make(3, 4), make(4, 5)])
    counter: typing.Counter[int] = collections.Counter()
    for i in range(100000):
        for sample in _spectrogram_model._random_nonoverlapping_alignments(alignments, 3):
            for i in range(sample.script[0], sample.script[1]):
                counter[i] += 1
    assert set(counter.keys()) == set(range(0, 5))
    assert_uniform_distribution(counter, abs=0.02)


def test__random_nonoverlapping_alignments__empty():
    """Test `_spectrogram_model._random_nonoverlapping_alignments` handles empty list. """
    assert _spectrogram_model._random_nonoverlapping_alignments(tuple(), 3) == tuple()


def test__random_nonoverlapping_alignments__large_max():
    """Test `_spectrogram_model._random_nonoverlapping_alignments` handles a large maximum. """
    make = lambda a, b: Alignment((a, b), (a, b), (a, b))
    with torchnlp.random.fork_rng(1234):
        alignments = tuple([make(0, 1), make(1, 2), make(2, 3), make(3, 4), make(4, 5)])
        assert len(_spectrogram_model._random_nonoverlapping_alignments(alignments, 1000000)) == 6


def test__get_loudness():
    """Test `_spectrogram_model._get_loudnes` slices, measures, and rounds loudness correctly. """
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
        loundess = _spectrogram_model._get_loudness(
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
    """Test `_spectrogram_model._get_char_to_word` maps characters to words correctly. """
    nlp = lib.text.load_en_english()
    doc = nlp("It was time to present the present abcdefghi.")
    char_to_word = _spectrogram_model._get_char_to_word(doc)
    expected = [0, 0, -1, 1, 1, 1, -1, 2, 2, 2, 2, -1, 3, 3, -1, 4, 4, 4, 4, 4, 4, 4, -1, 5, 5, 5]
    expected += [-1, 6, 6, 6, 6, 6, 6, 6, -1, 7, 7, 7, 7, 7, 7, 7, 7, 7, 8]
    assert char_to_word == expected


def test__get_word_vectors():
    """Test `_spectrogram_model._get_word_vectors` maps word vectors onto characters. """
    text = "It was time to present the present abcdefghi."
    doc = lib.text.load_en_core_web_md()(text)
    char_to_word = _spectrogram_model._get_char_to_word(doc)
    word_vectors = _spectrogram_model._get_word_vectors(char_to_word, doc)

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
    """ Test `_spectrogram_model._signals_to_spectrograms` is invariant to the batch size. """
    fft_length = 2048
    hop_length = fft_length // 4
    window = torch.from_numpy(librosa.filters.get_window("hann", fft_length)).float()
    batch_size = 16
    signals = []
    for i in range(batch_size):
        signal = lib.audio.pad_remainder(torch.randn(2400 + i * 300).numpy(), hop_length)
        signals.append(torch.from_numpy(signal))
    spectrogram, spectrogram_mask = _spectrogram_model._signals_to_spectrograms(
        signals, fft_length=fft_length, frame_hop=hop_length, window=window
    )
    module = lib.audio.SignalTodBMelSpectrogram(fft_length, hop_length, window=window)
    for i in range(batch_size):
        result = module(signals[i], aligned=True)
        expected = spectrogram.tensor[:, i][: spectrogram.lengths[:, i]]
        assert_almost_equal(expected, result, decimal=5)
        assert spectrogram_mask.tensor[:, i].sum() == spectrogram.lengths[:, i]
        assert spectrogram_mask.tensor[:, i].sum() == spectrogram_mask.lengths[:, i]


def test__get_normalized_half_gaussian():
    """Test `_spectrogram_model._get_normalized_half_gaussian` generates the left-side of a gaussian
    distribution normalized from 0 to 1."""
    expected_ = [0.0015184, 0.0070632, 0.0292409, 0.0952311, 0.2443498, 0.4946532, 0.7909854, 1.0]
    expected = torch.tensor(expected_)
    assert_almost_equal(_spectrogram_model._get_normalized_half_gaussian(8, 2), expected)


def test__make_stop_token():
    """ Test `_spectrogram_model._make_stop_token` makes a batched stop token. """
    spectrogram = SequenceBatch(torch.ones(8, 4, 16), torch.tensor([2, 4, 6, 8]).unsqueeze(0))
    stop_token = _spectrogram_model._make_stop_token(spectrogram, 6, 2)
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


def test_get_num_skipped():
    """ Test `_spectrogram_model.get_num_skipped` counts skipped tokens correctly. """
    alignments_ = [
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]],  # Test no skips
        [[1, 0, 0], [0, 1, 0], [0, 1, 0]],  # Test skipped
        [[1, 0, 0], [0, 1, 0], [0, 1, 0]],
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    ]
    alignments = torch.tensor(alignments_).transpose(0, 1).float()
    spectrogram_mask_ = [
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 0],  # Test that a masked frame is ignored
    ]
    spectrogram_mask = torch.tensor(spectrogram_mask_).transpose(0, 1).bool()
    token_mask_ = [
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 0],  # Test that a masked token cannot be skipped
        [1, 1, 1],
    ]
    token_mask = torch.tensor(token_mask_).transpose(0, 1).bool()
    num_skips = _spectrogram_model.get_num_skipped(alignments, token_mask, spectrogram_mask)
    assert num_skips.tolist() == [0.0, 1.0, 0.0, 1.0]


def test_get_num_skipped__zero_elements():
    """ Test `_spectrogram_model.get_num_skipped` handles zero elements correctly. """
    args = (
        torch.empty(1024, 0, 1024),
        torch.empty(1024, 0, dtype=torch.bool),
        torch.empty(1024, 0, dtype=torch.bool),
    )
    assert _spectrogram_model.get_num_skipped(*args).shape == (0,)


def _get_db_spectrogram(signal, **kwargs) -> torch.Tensor:
    spectrogram = torch.stft(signal.view(1, -1), **kwargs)
    spectrogram = torch.norm(spectrogram.double(), dim=-1)
    return lib.audio.amplitude_to_db(spectrogram).permute(2, 0, 1)


def test_get_cumulative_power_rms_level():
    """Test `_spectrogram_model.get_cumulative_power_rms_level` gets an approximate dB RMS level
    from a dB spectrogram."""
    frame_length = 1024
    frame_hop = frame_length // 4
    window = torch.ones(frame_length)
    _db_spectrogram = lambda s: _get_db_spectrogram(
        torch.tensor(s),
        n_fft=frame_length,
        hop_length=frame_hop,
        win_length=len(window),
        window=window,
        center=False,
    )
    db_spectrogram_ = [
        _db_spectrogram(lib.audio.full_scale_square_wave()),
        _db_spectrogram(lib.audio.full_scale_sine_wave()),
    ]
    db_spectrogram = torch.cat(db_spectrogram_, dim=1)
    rms = _spectrogram_model.get_cumulative_power_rms_level(db_spectrogram, window=window)
    assert_almost_equal(rms / db_spectrogram.shape[0], torch.Tensor([1.0000001, 0.500006]))


def test_get_cumulative_power_rms_level__precise():
    """Test `_spectrogram_model.get_cumulative_power_rms_level` gets an exact dB RMS level from a
    dB spectrogram."""
    frame_length = 1024
    frame_hop = frame_length // 4
    window = torch.ones(frame_length)
    sample_rate = 48000
    _db_spectrogram = lambda s: _get_db_spectrogram(
        lib.utils.pad_tensor(torch.tensor(s), (frame_length, frame_length)),
        n_fft=frame_length,
        hop_length=frame_hop,
        win_length=len(window),
        window=window,
        center=False,
    )
    db_spectrogram_ = [
        _db_spectrogram(lib.audio.full_scale_square_wave()),
        _db_spectrogram(lib.audio.full_scale_sine_wave()),
    ]
    db_spectrogram = torch.cat(db_spectrogram_, dim=1)
    rms = _spectrogram_model.get_cumulative_power_rms_level(db_spectrogram, window=window)
    assert_almost_equal(rms / (sample_rate / frame_hop), torch.Tensor([1.0, 0.49999998418]))


def test_get_average_db_rms_level():
    """Test `_spectrogram_model.get_cumulative_power_rms_level` gets the correct RMS level for
    a test file."""
    audio_path = TEST_DATA_PATH / "audio" / "bit(rate(lj_speech,24000),32).wav"
    metadata = lib.audio.get_audio_metadata(audio_path)
    lib.audio.assert_audio_normalized(metadata)
    audio = lib.audio.read_audio(audio_path)
    audio = _spectrogram_model._pad_and_trim_signal(audio)
    signal_to_spectrogram = lambda s, **k: _signals_to_spectrograms([s], **k)[0].tensor
    db_mel_spectrogram = signal_to_spectrogram(audio, get_weighting=lib.audio.identity_weighting)
    rms_level = _spectrogram_model.get_average_db_rms_level(db_mel_spectrogram).item()
    # NOTE: Audacity measured this RMS to be -23.6371. And `signal_to_rms` measured RMS to be
    # -23.6365.
    assert rms_level == -23.64263916015625

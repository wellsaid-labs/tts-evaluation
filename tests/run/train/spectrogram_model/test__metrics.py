import functools

import config as cf
import pytest
import torch

import lib
import run
from run._models.spectrogram_model import Preds
from run.train.spectrogram_model import _data, _metrics
from tests._utils import TEST_DATA_PATH, assert_almost_equal


@pytest.fixture(autouse=True, scope="module")
def run_around_tests():
    """Set a basic configuration."""
    run._config.configure()
    yield
    cf.purge()


def _make_preds(
    alignments: torch.Tensor, tokens_mask: torch.Tensor, frames_mask: torch.Tensor
) -> Preds:
    """Make `Preds` for computing metrics.

    Args:
        alignments (torch.FloatTensor [num_frames, batch_size, num_tokens])
        tokens_mask (torch.BoolTensor [batch_size, num_tokens])
        frames_mask (torch.BoolTensor [batch_size, num_frames])
    """
    return Preds(
        frames=torch.tensor(0),
        stop_tokens=torch.tensor(0),
        alignments=alignments,
        num_frames=frames_mask.sum(dim=1),
        frames_mask=frames_mask,
        num_tokens=tokens_mask.sum(dim=1),
        tokens_mask=tokens_mask,
        reached_max=torch.tensor(0),
    )


def test_get_num_skipped():
    """Test `_metrics.get_num_skipped` counts skipped tokens correctly."""
    alignments_ = [
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]],  # Test no skips
        [[1, 0, 0], [0, 1, 0], [0, 1, 0]],  # Test skipped
        [[1, 0, 0], [0, 1, 0], [0, 1, 0]],
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    ]
    alignments = torch.tensor(alignments_).transpose(0, 1).float()
    frames_mask_ = [
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 0],  # Test that a masked frame is ignored
    ]
    frames_mask = torch.tensor(frames_mask_).bool()
    tokens_mask_ = [
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 0],  # Test that a masked token cannot be skipped
        [1, 1, 1],
    ]
    tokens_mask = torch.tensor(tokens_mask_).bool()
    num_skips = _metrics.get_num_skipped(_make_preds(alignments, tokens_mask, frames_mask))
    assert num_skips.tolist() == [0.0, 1.0, 0.0, 1.0]


def test_get_num_skipped__zero_elements():
    """Test `_metrics.get_num_skipped` handles zero elements correctly."""
    preds = _make_preds(
        torch.empty(1024, 0, 1024),
        torch.empty(0, 1024, dtype=torch.bool),
        torch.empty(0, 1024, dtype=torch.bool),
    )
    assert _metrics.get_num_skipped(preds).shape == (0,)


def test_get_num_jumps():
    """Test `_metrics.get_num_jumps` counts jumps correctly."""
    alignments_ = [
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]],  # Test no jumps
        [[1, 0, 0], [0, 0, 1], [0, 0, 1]],  # Test one jump
        [[0, 0, 1], [1, 0, 0], [0, 0, 1]],  # Test three jumps, including backwards jumps
        [[1, 0, 0], [0, 0.25, 0.75], [0, 0.25, 0.75]],  # Test masked token with no jumps
        [[1, 0, 0], [1, 0, 0], [0, 0, 1]],  # Test masked frame with no jumps
    ]
    alignments = torch.tensor(alignments_).transpose(0, 1).float()
    frames_mask_ = [
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 0],  # Test that a masked frame is ignored
    ]
    frames_mask = torch.tensor(frames_mask_).bool()
    tokens_mask_ = [
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 0],  # Test that a masked token cannot be selected
        [1, 1, 1],
    ]
    tokens_mask = torch.tensor(tokens_mask_).bool()
    num_skips = _metrics.get_num_jumps(_make_preds(alignments, tokens_mask, frames_mask))
    assert num_skips.tolist() == [0.0, 1.0, 3.0, 0.0, 0.0]


def test_get_num_jumps__zero_elements():
    """Test `_metrics.get_num_jumps` handles zero elements correctly."""
    preds = _make_preds(
        torch.empty(1024, 0, 1024),
        torch.empty(0, 1024, dtype=torch.bool),
        torch.empty(0, 1024, dtype=torch.bool),
    )
    assert _metrics.get_num_jumps(preds).shape == (0,)


def test_get_num_small_max():
    """Test `_metrics.get_num_small_max` counts jumps correctly."""
    alignments_ = [
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]],  # Test non small focus
        [[1, 0, 0], [0, 0.9, 0.1], [0, 0.1, 0.9]],  # Test two small focus
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]],  # Test masked last token
        [[1, 0, 0], [0, 0.9, 0.1], [0, 0.1, 0.9]],  # Test masked last frame
    ]
    alignments = torch.tensor(alignments_).transpose(0, 1).float()
    frames_mask_ = [
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 0],  # Test that a masked frame is ignored
    ]
    frames_mask = torch.tensor(frames_mask_).bool()
    tokens_mask_ = [
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 0],  # Test that a masked token cannot be selected
        [1, 1, 1],
    ]
    tokens_mask = torch.tensor(tokens_mask_).bool()
    num_skips = _metrics.get_num_small_max(_make_preds(alignments, tokens_mask, frames_mask), 0.95)
    assert num_skips.tolist() == [0.0, 2.0, 1.0, 1.0]


def test_get_num_repeated():
    """Test `_metrics.get_num_repeated` counts repeated tokens correctly."""
    alignments_ = [
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]],  # Test no repeats
        [[1, 0, 0], [0, 1, 0], [0, 1, 0]],  # Test one repeated
        [[1, 0, 0], [0, 0, 1], [0, 0, 1]],  # Test masked last token (no repeats)
        [[1, 0, 0], [0, 1, 0], [0, 1, 0]],  # Test masked last frame (no repeats)
    ]
    alignments = torch.tensor(alignments_).transpose(0, 1).float()
    frames_mask_ = [
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 0],  # Test that a masked frame is ignored
    ]
    frames_mask = torch.tensor(frames_mask_).bool()
    tokens_mask_ = [
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 0],  # Test that a masked token cannot be selected
        [1, 1, 1],
    ]
    tokens_mask = torch.tensor(tokens_mask_).bool()
    num_skips = _metrics.get_num_repeated(_make_preds(alignments, tokens_mask, frames_mask), 1.0)
    assert num_skips.tolist() == [0.0, 1.0, 0.0, 0.0]


def _get_db_spectrogram(signal, **kwargs) -> torch.Tensor:
    spectrogram = torch.stft(signal.view(1, -1), **kwargs)
    spectrogram = torch.norm(spectrogram.double(), dim=-1)
    return lib.audio.amp_to_db(spectrogram).permute(2, 0, 1)


def test_get_power_rms_level_sum():
    """Test `_metrics.get_power_rms_level_sum` gets an approximate dB RMS level
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
    rms = _metrics.get_power_rms_level_sum(
        db_spectrogram, window=window, window_correction_factor=None
    )
    assert_almost_equal(rms / db_spectrogram.shape[0], torch.Tensor([1.0000001, 0.500006]))


def test_get_power_rms_level_sum__precise():
    """Test `_metrics.get_power_rms_level_sum` gets an exact dB RMS level from a dB spectrogram."""
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
    rms = _metrics.get_power_rms_level_sum(
        db_spectrogram, window=window, window_correction_factor=None
    )
    assert_almost_equal(rms / (sample_rate / frame_hop), torch.Tensor([1.0, 0.49999998418]))


def test_get_average_db_rms_level():
    """Test `_metrics.get_power_rms_level_sum` gets the correct RMS level for a test file."""
    audio_path = TEST_DATA_PATH / "audio" / "bit(rate(lj_speech,24000),32).wav"
    metadata = lib.audio.get_audio_metadata(audio_path)
    run.data._loader.is_normalized_audio_file(metadata, **cf.get())
    audio = lib.audio.read_audio(audio_path)
    audio = _data._pad_and_trim_signal(audio)
    signal_to_spectrogram = lambda s, **k: _data._signals_to_spectrograms([s], **k)[0].tensor
    db_mel_spectrogram = signal_to_spectrogram(audio, get_weighting=lib.audio.identity_weighting)
    rms_level = _metrics.get_average_db_rms_level(db_mel_spectrogram).item()
    # NOTE: Audacity measured this RMS to be -23.6371. And `signal_to_rms` measured RMS to be
    # -23.6365.
    assert rms_level == pytest.approx(-23.64263916015625, abs=0.001)


def test_get_num_pause_frames():
    """Test `_metrics.get_power_rms_level_sum` gets the correct number of pause frames."""
    audio_path = TEST_DATA_PATH / "audio" / "bit(rate(lj_speech,24000),32).wav"
    metadata = lib.audio.get_audio_metadata(audio_path)
    run.data._loader.is_normalized_audio_file(metadata, **cf.get())
    audio = lib.audio.read_audio(audio_path)
    fft_length = 2048
    frame_hop = fft_length // 4
    sample_rate = metadata.sample_rate
    window = run._utils.get_window("hann", fft_length, frame_hop)
    audio = torch.tensor(lib.audio.pad_remainder(audio, multiple=frame_hop))
    signal_to_spectrogram = lambda s, **k: _data._signals_to_spectrograms([s], **k)[0].tensor
    db_mel_spectrogram = signal_to_spectrogram(
        audio,
        get_weighting=lib.audio.iso226_weighting,
        frame_hop=frame_hop,
        fft_length=fft_length,
        sample_rate=sample_rate,
        window=window,
        min_decibel=-50,
        num_mel_bins=128,
    )
    get_num_pause_frames = functools.partial(
        _metrics.get_num_pause_frames, frame_hop=frame_hop, sample_rate=sample_rate, window=window
    )
    assert get_num_pause_frames(db_mel_spectrogram, None, -40.0, frame_hop / sample_rate) == [97]
    assert get_num_pause_frames(db_mel_spectrogram, None, -40.0, 0.25) == [38]
    # NOTE: Test `max_loudness` is too quiet.
    assert get_num_pause_frames(db_mel_spectrogram, None, -80.0, frame_hop / sample_rate) == [0]
    # NOTE: Test `min_length` is too long.
    assert get_num_pause_frames(db_mel_spectrogram, None, -40.0, 1) == [0]
    mask = torch.zeros(*db_mel_spectrogram.shape[:2]).transpose(0, 1)
    assert get_num_pause_frames(db_mel_spectrogram, mask, -40.0, frame_hop / sample_rate) == [0]
    batch = torch.cat([db_mel_spectrogram, db_mel_spectrogram], dim=1)
    assert get_num_pause_frames(batch, None, -40.0, frame_hop / sample_rate) == [97, 97]

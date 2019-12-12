import io

from scipy.io import wavfile

import librosa
import numpy as np
import pytest
import torch

from src.audio import build_wav_header
from src.audio import cache_get_audio_metadata
from src.audio import combine_signal
from src.audio import get_audio_metadata
from src.audio import get_log_mel_spectrogram
from src.audio import get_num_seconds
from src.audio import griffin_lim
from src.audio import normalize_audio
from src.audio import read_audio
from src.audio import split_signal
from src.audio import write_audio
from src.environment import TEST_DATA_PATH

TEST_DATA_PATH_LOCAL = TEST_DATA_PATH / 'test_audio'


def test_get_num_seconds():
    """ Ensure `get_num_seconds` works regardless of the audio metadata like sample rate.

    TODO: The number of seconds for both files is not exactly the same. Look into that.
    """
    assert pytest.approx(7.5848, 0.0001) == get_num_seconds(TEST_DATA_PATH_LOCAL / 'lj_speech.wav')
    assert pytest.approx(7.5848, 0.0001) == get_num_seconds(TEST_DATA_PATH_LOCAL /
                                                            'rate(lj_speech,24000).wav')


def test_read_audio():
    path = TEST_DATA_PATH_LOCAL / 'rate(lj_speech,24000).wav'

    integer = read_audio(path, to_float=False)
    assert integer.dtype == np.dtype('int16')

    float_ = read_audio(path, to_float=True)
    assert float_.dtype == np.dtype('float16')

    expected = (integer / 2**15).astype(np.dtype('float16'))

    np.testing.assert_almost_equal(expected, float_)

    librosa_, _ = librosa.core.load(path, sr=None, mono=False)

    np.testing.assert_almost_equal(librosa_.astype(np.dtype('float16')), float_)


def test_write_audio():
    filename = TEST_DATA_PATH_LOCAL / 'lj_speech.wav'
    metadata = get_audio_metadata(filename)
    sample_rate, signal = wavfile.read(str(filename))

    new_filename = TEST_DATA_PATH_LOCAL / 'lj_speech_two.wav'
    write_audio(new_filename, signal, sample_rate)
    new_metadata = get_audio_metadata(new_filename)

    assert metadata == new_metadata  # Ensure the metadata stays the same


def test_write_audio__read_audio():
    filename = TEST_DATA_PATH_LOCAL / 'rate(lj_speech,24000).wav'
    metadata = get_audio_metadata(filename)
    signal = read_audio(str(filename), to_float=False)

    new_filename = TEST_DATA_PATH_LOCAL / 'clone(rate(lj_speech,24000)).wav'
    write_audio(new_filename, signal, 24000)
    new_metadata = get_audio_metadata(new_filename)

    assert metadata == new_metadata  # Ensure the metadata stays the same


def test_cache_get_audio_metadata():
    path = TEST_DATA_PATH_LOCAL / 'rate(lj_speech,24000).wav'
    cache_get_audio_metadata([path])
    assert get_audio_metadata.cache.get(kwargs={'audio_path': path}) == {
        'sample_rate': 24000,
        'bits': 16,
        'channels': 1,
        'encoding': 'signed-integer'
    }


def test_get_audio_metadata():
    path = TEST_DATA_PATH_LOCAL / 'rate(lj_speech,24000).wav'
    assert {
        'sample_rate': 24000,
        'bits': 16,
        'channels': 1,
        'encoding': 'signed-integer'
    } == get_audio_metadata(path)


def test_build_wav_header():
    sample_rate = 16000
    file_ = io.BytesIO()
    wavfile.write(file_, sample_rate, np.int16([]))
    expected_header = file_.getvalue()
    wav_header, header_length = build_wav_header(0, sample_rate)
    assert len(wav_header) == header_length
    assert expected_header == wav_header


def test_log_mel_spectrogram_smoke():
    """ Smoke test to ensure everything runs.
    """
    frame_size = 1200
    frame_hop = 300
    path = TEST_DATA_PATH_LOCAL / 'rate(lj_speech,24000).wav'
    sample_rate = 24000
    signal = read_audio(path, {
        'sample_rate': sample_rate,
        'bits': 16,
        'channels': 1,
        'encoding': 'signed-integer'
    })
    log_mel_spectrogram, padding = get_log_mel_spectrogram(
        signal, sample_rate, frame_size=frame_size, frame_hop=frame_hop)

    assert log_mel_spectrogram.dtype == np.float32
    assert len(log_mel_spectrogram.shape) == 2
    assert len(signal.shape) == 1
    assert int(signal.shape[0] + sum(padding)) / int(log_mel_spectrogram.shape[0]) == frame_hop


def test_griffin_lim_smoke():
    """ Smoke test to ensure everything runs.
    """
    path = TEST_DATA_PATH_LOCAL / 'rate(lj_speech,24000).wav'
    sample_rate = 24000
    signal = read_audio(path, {
        'sample_rate': sample_rate,
        'channels': 1,
        'bits': 16,
        'encoding': 'signed-integer',
    })
    log_mel_spectrogram, _ = get_log_mel_spectrogram(signal, sample_rate)
    waveform = griffin_lim(log_mel_spectrogram, sample_rate)
    assert len(waveform) > 0


def test_split_signal():
    signal = torch.FloatTensor([1.0, -1.0, 0, 2**-7, 2**-8])
    coarse, fine = split_signal(signal, 16)
    assert torch.equal(coarse, torch.LongTensor([255, 0, 128, 129, 128]))
    assert torch.equal(fine, torch.LongTensor([255, 0, 0, 0, 2**7]))


def test_combine_signal_return_int():
    signal = torch.FloatTensor([1.0, -1.0, 0, 2**-7, 2**-8])
    coarse, fine = split_signal(signal, 16)
    new_signal = combine_signal(coarse, fine, 16, return_int=True)
    expected_signal = torch.IntTensor([2**15 - 1, -2**15, 0, 256, 128])
    np.testing.assert_allclose(expected_signal.numpy(), new_signal.numpy())


def test_combine_signal():
    signal = torch.FloatTensor([1.0, -1.0, 0, 2**-7, 2**-8])
    coarse, fine = split_signal(signal, 16)
    new_signal = combine_signal(coarse, fine, 16)
    # NOTE: 1.0 gets clipped to ``(2**15 - 1) / 2**15``
    expected_signal = torch.FloatTensor([(2**15 - 1) / 2**15, -1.0, 0, 2**-7, 2**-8])
    np.testing.assert_allclose(expected_signal.numpy(), new_signal.numpy())


def test_split_combine_signal():
    signal = torch.FloatTensor(1000).uniform_(-1.0, 1.0)
    reconstructed_signal = combine_signal(*split_signal(signal), bits=16)
    np.testing.assert_allclose(signal.numpy(), reconstructed_signal.numpy(), atol=1e-03)


def test_split_combine_signal_multiple_dim():
    signal = torch.FloatTensor(1000, 1000).uniform_(-1.0, 1.0)
    reconstructed_signal = combine_signal(*split_signal(signal), bits=16)
    np.testing.assert_allclose(signal.numpy(), reconstructed_signal.numpy(), atol=1e-03)


def test_normalize_audio():
    path = TEST_DATA_PATH_LOCAL / 'lj_speech.wav'
    new_audio_path = normalize_audio(
        path, bits=8, sample_rate=24000, channels=2, encoding='unsigned-integer')
    assert (new_audio_path.stem ==
            'encoding(channels(bits(rate(lj_speech,24000),8),2),unsigned-integer)')
    assert new_audio_path.exists()
    assert {
        'bits': 8,
        'sample_rate': 24000,
        'channels': 2,
        'encoding': 'unsigned-integer'
    } == get_audio_metadata(new_audio_path)


def test_normalize_audio__not_normalized():
    path = TEST_DATA_PATH_LOCAL / 'rate(lj_speech,24000).wav'
    normalized_audio_path = normalize_audio(
        path, bits=16, sample_rate=24000, channels=1, encoding='signed-integer')
    assert path == normalized_audio_path

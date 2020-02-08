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
from src.audio import MultiResolutionMelSpectrogramLoss
from src.audio import normalize_audio
from src.audio import read_audio
from src.audio import SignalToLogMelSpectrogram
from src.audio import split_signal
from src.audio import write_audio
from src.environment import DATA_PATH
from src.environment import TEST_DATA_PATH

TEST_DATA_PATH_LOCAL = TEST_DATA_PATH / 'test_audio'


def test_signal_to_log_mel_spectrogram():
    path = TEST_DATA_PATH_LOCAL / 'rate(lj_speech,24000).wav'
    sample_rate = 24000
    signal = read_audio(path, {
        'sample_rate': sample_rate,
        'bits': 16,
        'channels': 1,
        'encoding': 'signed-integer'
    })
    log_mel_spectrogram = get_log_mel_spectrogram(signal, sample_rate, center=False)
    module = SignalToLogMelSpectrogram(sample_rate=sample_rate)
    other_log_mel_spectrogram = module(torch.tensor(signal)).detach().numpy()
    assert log_mel_spectrogram.shape == other_log_mel_spectrogram.shape
    # NOTE: Numpy does it's computations in 64-bit while PyTorch does it's in 32-bit. This causes
    # some error.
    # NOTE: Numpy can be more accurate than PyTorch, for example:
    # https://github.com/pytorch/pytorch/issues/19164
    np.testing.assert_almost_equal(log_mel_spectrogram, other_log_mel_spectrogram, decimal=3)


def test_signal_to_log_mel_spectrogram_backward():
    """ Ensure `SignalToLogMelSpectrogram` is differentiable. """
    tensor = torch.nn.Parameter(torch.randn(2400))
    module = SignalToLogMelSpectrogram()
    module(tensor).sum().backward()


def test_signal_to_log_mel_spectrogram_batch_invariance():
    """ Ensure `SignalToLogMelSpectrogram` is batch invariant. """
    tensor = torch.nn.Parameter(torch.randn(10, 2400))
    module = SignalToLogMelSpectrogram()
    results = module(tensor)
    result = module(tensor[0])
    np.testing.assert_almost_equal(results[0].detach().numpy(), result.detach().numpy(), decimal=5)


def test_get_num_seconds():
    """ Ensure `get_num_seconds` works regardless of the audio metadata like sample rate.

    TODO: The number of seconds for both files is not exactly the same. Look into that.
    """
    assert pytest.approx(7.5839, 0.0001) == get_num_seconds(TEST_DATA_PATH_LOCAL / 'lj_speech.wav')
    assert pytest.approx(7.5839, 0.0001) == get_num_seconds(TEST_DATA_PATH_LOCAL /
                                                            'rate(lj_speech,24000).wav')
    assert pytest.approx(2.070, 0.0001) == get_num_seconds(
        DATA_PATH / 'M-AILABS/en_US/by_book/female/judy_bieber/' /
        'dorothy_and_wizard_oz/wavs/dorothy_and_wizard_oz_01_f000001.wav')


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
    fft_length = 2048
    signal = read_audio(path, {
        'sample_rate': sample_rate,
        'bits': 16,
        'channels': 1,
        'encoding': 'signed-integer'
    })
    log_mel_spectrogram, padded_signal = get_log_mel_spectrogram(
        signal, sample_rate, frame_size=frame_size, frame_hop=frame_hop, fft_length=fft_length)

    assert log_mel_spectrogram.dtype == np.float32
    assert len(log_mel_spectrogram.shape) == 2
    assert len(padded_signal.shape) == 1
    assert int(padded_signal.shape[0]) / int(log_mel_spectrogram.shape[0]) == frame_hop


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


def test_griffin_lim_large_numbers():
    """ Test if large numbers will error out griffin lim.
    """
    griffin_lim(np.random.uniform(low=-200, high=200, size=(50, 50)), 24000)


def test_griffin_lim_small_size():
    """ Test if small array will error out griffin lim.
    """
    griffin_lim(np.random.uniform(low=-1, high=1, size=(1, 1)), 24000)


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


def test_multi_resolution_stft_loss_griffin_lim():
    path = TEST_DATA_PATH_LOCAL / 'rate(lj_speech,24000).wav'
    sample_rate = 24000
    signal = read_audio(path, {
        'sample_rate': sample_rate,
        'bits': 16,
        'channels': 1,
        'encoding': 'signed-integer'
    })
    criterion = MultiResolutionMelSpectrogramLoss()

    log_mel_spectrogram, signal = get_log_mel_spectrogram(signal, sample_rate=sample_rate)
    worse_waveform = griffin_lim(log_mel_spectrogram, sample_rate, iterations=5)
    better_waveform = griffin_lim(log_mel_spectrogram, sample_rate, iterations=15)

    spectral_convergence_loss, log_mel_spectrogram_magnitude_loss = criterion(
        torch.tensor(worse_waveform), torch.tensor(signal))
    better_spectral_convergence_loss, better_log_mel_spectrogram_magnitude_loss = criterion(
        torch.tensor(better_waveform), torch.tensor(signal))

    assert log_mel_spectrogram_magnitude_loss > better_log_mel_spectrogram_magnitude_loss
    assert spectral_convergence_loss > better_spectral_convergence_loss


def test_multi_resolution_stft_loss_zero():
    signal = torch.normal(0, 1, size=(5, 3600))
    criterion = MultiResolutionMelSpectrogramLoss()
    spectral_convergence_loss, log_stft_magnitude_loss = criterion(signal, signal)
    assert spectral_convergence_loss == 0
    assert log_stft_magnitude_loss == 0


def test_multi_resolution_stft_loss_batch_invariance():
    signal = torch.normal(0, 1, size=(1, 3600))
    target = torch.normal(0, 1, size=(1, 3600))

    batched_signal = torch.cat([signal, signal, signal])
    batched_target = torch.cat([target, target, target])

    criterion = MultiResolutionMelSpectrogramLoss()

    spectral_convergence_loss, log_stft_magnitude_loss = criterion(signal, target)
    batched_spectral_convergence_loss, batched_log_stft_magnitude_loss = criterion(
        batched_signal, batched_target)

    assert_almost_equal = lambda a, b: np.testing.assert_almost_equal(
        a.detach().numpy(), b.detach().numpy(), decimal=5)

    assert_almost_equal(spectral_convergence_loss, batched_spectral_convergence_loss)
    assert_almost_equal(log_stft_magnitude_loss, batched_log_stft_magnitude_loss)

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
from src.audio import integer_to_floating_point_pcm
from src.audio import iso226_weighting
from src.audio import normalize_audio
from src.audio import read_audio
from src.audio import SignalToLogMelSpectrogram
from src.audio import split_signal
from src.audio import WavFileMetadata
from src.audio import write_audio
from src.environment import DATA_PATH
from src.environment import TEST_DATA_PATH
from src.utils import make_arg_key
from src.audio import power_to_db
from src.audio import db_to_power
from src.audio import amplitude_to_db
from src.audio import db_to_amplitude

TEST_DATA_PATH_LOCAL = TEST_DATA_PATH / 'test_audio'


def test_amplitude_to_db():
    t = torch.abs(torch.randn(100))
    np.testing.assert_almost_equal(db_to_amplitude(amplitude_to_db(t)).numpy(), t.numpy())


def test_power_to_db():
    t = torch.abs(torch.randn(100))
    np.testing.assert_almost_equal(db_to_power(power_to_db(t)).numpy(), t.numpy())


def test_iso226_weighting():
    """ Test ISO226 weighting based off these charts:
    https://en.wikipedia.org/wiki/A-weighting#/media/File:Lindos3.svg
    https://github.com/wellsaid-labs/Text-to-Speech/blob/adc1f28864c9515a5d33b876b135d2e95da73faf/weights.png
    """
    np.testing.assert_almost_equal([-59.843877, 0.0, -59.843877],
                                   iso226_weighting(np.array([20, 1000, 20000])))


def test_signal_to_log_mel_spectrogram_against_librosa():
    n_fft = 2048
    win_length = 2048
    hop_length = 512
    amin = 1e-10
    n_mels = 128
    min_decibel = -50.0
    sample_rate = 24000

    path = TEST_DATA_PATH_LOCAL / 'rate(lj_speech,24000).wav'
    metadata = WavFileMetadata(sample_rate, 16, 1, 'signed-integer')
    signal = integer_to_floating_point_pcm(read_audio(path, metadata))

    # Learn more about this algorithm here:
    # https://github.com/librosa/librosa/issues/463#issuecomment-265165830
    S = librosa.stft(
        signal,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        center=False,
        window='hann')
    S = np.abs(S).astype(np.float32)**2.0
    log_S = librosa.perceptual_weighting(
        S, librosa.fft_frequencies(sr=sample_rate, n_fft=n_fft), amin=amin,
        top_db=None).astype(np.float32)
    melspec = librosa.feature.melspectrogram(
        S=librosa.db_to_power(log_S),
        sr=sample_rate,
        n_fft=n_fft,
        htk=True,
        norm=None,
        fmax=None,
        fmin=0.0)
    melspec = librosa.power_to_db(melspec, amin=amin, top_db=None)
    melspec = np.maximum(melspec, min_decibel).transpose()

    module = SignalToLogMelSpectrogram(
        fft_length=n_fft,
        frame_hop=hop_length,
        sample_rate=sample_rate,
        num_mel_bins=n_mels,
        window=torch.hann_window(win_length),
        min_decibel=min_decibel,
        get_weighting=librosa.A_weighting,
        lower_hertz=0,
        eps=amin)

    other_mel_spectrogram = module(torch.tensor(signal)).detach().numpy()
    assert melspec.shape == other_mel_spectrogram.shape
    np.testing.assert_almost_equal(melspec, other_mel_spectrogram, decimal=2)


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


def test_signal_to_log_mel_spectrogram_intermediate():
    """ Ensure `SignalToLogMelSpectrogram` is can returns intermediate values. """
    batch_size = 10
    n_fft = 2048
    tensor = torch.nn.Parameter(torch.randn(batch_size, 2400))
    module = SignalToLogMelSpectrogram(fft_length=n_fft)
    db_mel_spectrogram, db_spectrogram, spectrogram = module(tensor, intermediate=True)
    assert spectrogram.shape == (batch_size, db_mel_spectrogram.shape[1], n_fft // 2 + 1)
    assert db_spectrogram.shape == (batch_size, db_mel_spectrogram.shape[1], n_fft // 2 + 1)


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
    integer = read_audio(path)
    assert integer.dtype == np.int16


def test_integer_to_floating_point_pcm():
    int16 = np.random.randint(-2**15, 2**15, size=10000, dtype=np.int16)
    int32 = np.random.randint(-2**31, 2**31, size=10000, dtype=np.int32)

    np.testing.assert_almost_equal(
        integer_to_floating_point_pcm(int32), (int32.astype(np.float32) / 2**31))
    np.testing.assert_almost_equal(
        integer_to_floating_point_pcm(int16), (int16.astype(np.float32) / 2**15))

    assert integer_to_floating_point_pcm(int32).max() <= 1.0
    assert integer_to_floating_point_pcm(int32).min() >= -1.0

    assert integer_to_floating_point_pcm(int16).max() <= 1.0
    assert integer_to_floating_point_pcm(int16).min() >= -1.0


def test_integer_to_floating_point_pcm__real_audio():
    path = TEST_DATA_PATH_LOCAL / 'rate(lj_speech,24000).wav'
    integer = read_audio(path)
    assert integer.dtype == np.int16
    floating = integer_to_floating_point_pcm(integer)
    assert floating.dtype == np.float32
    expected = integer.astype(np.float32) / 2**15
    np.testing.assert_almost_equal(expected, floating)


def test_write_audio__invalid():
    # Ensure invalid audio files cannot be written
    with pytest.raises(AssertionError):
        write_audio('invalid.wav', np.array([0.0, 0.0, 0.5, -0.5], dtype=np.float64), 24000)

    with pytest.raises(AssertionError):
        write_audio('invalid.wav', np.array([1.1, 1.2, -3.0, -2.0], dtype=np.float32), 24000)

    # Ensure files cannot be overwritten
    with pytest.raises(ValueError):
        write_audio(TEST_DATA_PATH_LOCAL / 'lj_speech.wav',
                    np.array([0.0, 0.0, 0.0], dtype=np.float32), 24000)

    with pytest.raises(ValueError):
        write_audio(
            str(TEST_DATA_PATH_LOCAL / 'lj_speech.wav'), np.array([0.0, 0.0, 0.0],
                                                                  dtype=np.float32), 24000)


def test_write_audio():
    filename = TEST_DATA_PATH_LOCAL / 'lj_speech.wav'
    metadata = get_audio_metadata(filename)
    sample_rate, signal = wavfile.read(str(filename))

    new_filename = TEST_DATA_PATH_LOCAL / 'lj_speech_two.wav'
    write_audio(new_filename, signal, sample_rate)
    new_metadata = get_audio_metadata(new_filename)

    assert metadata == new_metadata  # Ensure the metadata stays the same


def test_read_audio_and_write_audio():
    sample_rate = 24000
    filename = 'test_read_audio_and_write_audio.wav'
    uint8 = np.random.randint(0, 256, size=10000, dtype=np.uint8)
    int16 = np.random.randint(-2**15, 2**15, size=10000, dtype=np.int16)
    int32 = np.random.randint(-2**31, 2**31, size=10000, dtype=np.int32)
    float32 = np.clip(np.random.normal(size=10000), -1.0, 1.0).astype(np.float32)
    encodings = ['unsigned-integer', 'signed-integer', 'signed-integer', 'floating-point']

    for signal, encoding in zip([uint8, int16, int32, float32], encodings):
        path = TEST_DATA_PATH_LOCAL / (str(signal.dtype) + '_' + filename)
        metadata = WavFileMetadata(sample_rate, signal.dtype.itemsize * 8, 1, encoding)
        write_audio(path, signal, sample_rate)
        loaded_signal = read_audio(path, metadata)
        np.testing.assert_almost_equal(signal, loaded_signal)


def test_write_audio_and_read_audio__real_file():
    filename = TEST_DATA_PATH_LOCAL / 'rate(lj_speech,24000).wav'
    metadata = get_audio_metadata(filename)
    signal = read_audio(str(filename))

    new_filename = TEST_DATA_PATH_LOCAL / 'clone(rate(lj_speech,24000)).wav'
    write_audio(new_filename, signal, 24000)
    new_metadata = get_audio_metadata(new_filename)

    assert metadata == new_metadata  # Ensure the metadata stays the same


def test_cache_get_audio_metadata():
    path = TEST_DATA_PATH_LOCAL / 'rate(lj_speech,24000).wav'
    cache_get_audio_metadata([path])
    assert get_audio_metadata.disk_cache.get(
        make_arg_key(get_audio_metadata.__wrapped__.__wrapped__,
                     path)) == WavFileMetadata(24000, 16, 1, 'signed-integer')


def test_get_audio_metadata():
    path = TEST_DATA_PATH_LOCAL / 'rate(lj_speech,24000).wav'
    assert get_audio_metadata(path) == WavFileMetadata(24000, 16, 1, 'signed-integer')


def test_build_wav_header():
    sample_rate = 16000
    file_ = io.BytesIO()
    wavfile.write(file_, sample_rate, np.int16([]))
    expected_header = file_.getvalue()
    wav_header, header_length = build_wav_header(0, sample_rate)
    assert len(wav_header) == header_length
    assert expected_header == wav_header


def test_log_mel_spectrogram():
    """ Smoke test to ensure everything runs.
    """
    frame_size = 1200
    frame_hop = 300
    path = TEST_DATA_PATH_LOCAL / 'rate(lj_speech,24000).wav'
    sample_rate = 24000
    fft_length = 2048
    signal = read_audio(path, WavFileMetadata(24000, 16, 1, 'signed-integer'))
    log_mel_spectrogram, padded_signal = get_log_mel_spectrogram(
        signal,
        sample_rate=sample_rate,
        window=torch.hann_window(frame_size),
        frame_hop=frame_hop,
        fft_length=fft_length)

    assert log_mel_spectrogram.dtype == np.float32
    assert len(log_mel_spectrogram.shape) == 2
    assert len(padded_signal.shape) == 1
    assert int(padded_signal.shape[0]) / int(log_mel_spectrogram.shape[0]) == frame_hop


def test_griffin_lim_smoke():
    """ Smoke test to ensure everything runs.
    """
    path = TEST_DATA_PATH_LOCAL / 'rate(lj_speech,24000).wav'
    sample_rate = 24000
    signal = read_audio(path, WavFileMetadata(24000, 16, 1, 'signed-integer'))
    log_mel_spectrogram, _ = get_log_mel_spectrogram(signal, sample_rate=sample_rate)
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
    assert get_audio_metadata(new_audio_path) == WavFileMetadata(24000, 8, 2, 'unsigned-integer')


def test_normalize_audio__not_normalized():
    path = TEST_DATA_PATH_LOCAL / 'rate(lj_speech,24000).wav'
    normalized_audio_path = normalize_audio(
        path, bits=16, sample_rate=24000, channels=1, encoding='signed-integer')
    assert path == normalized_audio_path

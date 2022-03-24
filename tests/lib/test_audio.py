import itertools
import pathlib
import random
import shutil
import subprocess
import tempfile
import typing
from functools import partial

import config as cf
import librosa
import numpy as np
import pytest
import torch
import torch.nn
from torch.nn import functional
from torchnlp.encoders.text import stack_and_pad_tensors

import lib
from lib.audio import (
    SignalTodBMelSpectrogram,
    amp_to_power,
    db_to_power,
    framed_rms_to_rms,
    power_spectrogram_to_framed_rms,
    power_to_db,
)
from tests import _utils

TEST_DATA_PATH = _utils.TEST_DATA_PATH / "audio"
TEST_DATA_LJ = TEST_DATA_PATH / "bit(rate(lj_speech,24000),32).wav"
TEST_DATA_LJ_MULTI_CHANNEL = TEST_DATA_PATH / "channels(bit(rate(lj_speech,24000),32),2).wav"


@pytest.fixture(scope="module", autouse=True)
def run_around_tests():
    """Set a basic configuration for `lib.audio`."""
    fft_length = 2048
    sample_rate = 24000
    num_mel_bins = 128
    assert fft_length % 4 == 0
    frame_hop = fft_length // 4
    window = librosa.filters.get_window("hann", fft_length)
    hertz_bounds = {"lower_hertz": 20, "upper_hertz": 20000}
    config = {
        power_spectrogram_to_framed_rms: cf.Args(window=torch.tensor(window).float()),
        lib.audio.signal_to_framed_rms: cf.Args(frame_length=fft_length, hop_length=frame_hop),
        lib.audio.pad_remainder: cf.Args(multiple=frame_hop, mode="constant", constant_values=0.0),
        lib.audio.write_audio: cf.Args(sample_rate=sample_rate),
        SignalTodBMelSpectrogram: cf.Args(
            sample_rate=sample_rate,
            frame_hop=frame_hop,
            window=torch.tensor(window).float(),
            fft_length=fft_length,
            num_mel_bins=num_mel_bins,
            min_decibel=-50.0,
            get_weighting=lib.audio.iso226_weighting,
            **hertz_bounds,
        ),
        lib.audio.griffin_lim: cf.Args(
            frame_hop=frame_hop,
            fft_length=fft_length,
            window=window,
            sample_rate=sample_rate,
            power=1.20,
            iterations=30,
            get_weighting=lib.audio.iso226_weighting,
            **hertz_bounds,
        ),
    }
    cf.add(config)
    yield
    cf.purge()


def test_unit_conversion():
    """Test unit conversion between seconds, milliseconds, and samples is self consistent."""
    rate = 8000
    for i in [0, 0.9, 1, 1.1]:
        assert lib.audio.sec_to_milli(lib.audio.milli_to_sec(i)) == i
        assert lib.audio.sample_to_milli(lib.audio.milli_to_sample(i * 10, rate), rate) == i * 10
        assert lib.audio.sample_to_sec(lib.audio.sec_to_sample(i, rate), rate) == i


def test__parse_audio_metadata():
    """Test `lib.audio._parse_audio_metadata` parses the metadata correctly."""
    metadata = lib.audio._parse_audio_metadata(
        """Input File     : 'data/Heather Doe/03 Recordings/Heather_4-21.wav'
    Channels       : 1
    Sample Rate    : 44100
    Precision      : 24-bit
    Duration       : 03:46:28.09 = 599234761 samples = 1.01911e+06 CDDA sectors
    File Size      : 1.80G
    Bit Rate       : 1.06M
    Sample Encoding: 24-bit Signed Integer PCM"""
    )
    assert metadata == lib.audio.AudioMetadata(
        path=pathlib.Path("data/Heather Doe/03 Recordings/Heather_4-21.wav"),
        sample_rate=44100,
        num_channels=1,
        encoding=lib.audio.AudioEncoding.PCM_INT_24_BIT,
        bit_rate="1.06M",
        precision="24-bit",
        num_samples=599234761,
    )


def test__parse_audio_metadata__mp3():
    """Test `lib.audio._parse_audio_metadata` parses the metadata, with a comment, correctly."""
    metadata = lib.audio._parse_audio_metadata(
        """Input File     : 'data/beth_cameron/recordings/4.mp3'
    Channels       : 1
    Sample Rate    : 44100
    Precision      : 16-bit
    Duration       : 00:12:18.62 = 32573274 samples = 55396.7 CDDA sectors
    File Size      : 23.6M
    Bit Rate       : 256k
    Sample Encoding: MPEG audio (layer I, II or III)
    Comment        : 'Title=WellSaid_Script4'"""
    )
    assert metadata == lib.audio.AudioMetadata(
        path=pathlib.Path("data/beth_cameron/recordings/4.mp3"),
        sample_rate=44100,
        num_channels=1,
        encoding=lib.audio.AudioEncoding.MPEG,
        bit_rate="256k",
        precision="16-bit",
        num_samples=32573274,
    )


def test__parse_audio_metadata__multiline_comments():
    """Test `lib.audio._parse_audio_metadata` parses the metadata, with a multiline comment."""
    metadata = lib.audio._parse_audio_metadata(
        """Input File     : 'coldcomforthvac_061915.mp3'
    Channels       : 1
    Sample Rate    : 44100
    Precision      : 16-bit
    Duration       : 00:04:57.33 = 13112253 samples = 22299.8 CDDA sectors
    File Size      : 9.51M
    Bit Rate       : 256k
    Sample Encoding: MPEG audio (layer I, II or III)
    Comments       :
    Year=2015
    Genre=0"""
    )
    assert metadata == lib.audio.AudioMetadata(
        path=pathlib.Path("coldcomforthvac_061915.mp3"),
        sample_rate=44100,
        num_channels=1,
        encoding=lib.audio.AudioEncoding.MPEG,
        bit_rate="256k",
        precision="16-bit",
        num_samples=13112253,
    )


def test_get_audio_metadata():
    """Test `lib.audio.get_audio_metadata` returns the right metadata."""
    assert lib.audio.get_audio_metadata(TEST_DATA_LJ) == lib.audio.AudioMetadata(
        path=TEST_DATA_LJ,
        sample_rate=24000,
        num_channels=1,
        encoding=lib.audio.AudioEncoding.PCM_FLOAT_32_BIT,
        bit_rate="768k",
        precision="25-bit",
        num_samples=182015,
    )


def test_get_audio_metadata__empty():
    """Test `lib.audio.get_audio_metadata` handles an empty list."""
    assert lib.audio.get_audio_metadata([]) == []


def test_get_audio_metadata__large_batch():
    """Test `lib.audio.get_audio_metadata` handles a large batch."""
    metadatas = lib.audio.get_audio_metadata([TEST_DATA_LJ] * 100, max_arg_length=124)
    for metadata in metadatas:
        assert metadata == lib.audio.AudioMetadata(
            path=TEST_DATA_LJ,
            sample_rate=24000,
            num_channels=1,
            encoding=lib.audio.AudioEncoding.PCM_FLOAT_32_BIT,
            bit_rate="768k",
            precision="25-bit",
            num_samples=182015,
        )


def test_get_audio_metadata__multiple_channels():
    """Test `lib.audio.get_audio_metadata` returns the right metadata for multiple channel audio."""
    assert lib.audio.get_audio_metadata(TEST_DATA_LJ_MULTI_CHANNEL) == lib.audio.AudioMetadata(
        path=TEST_DATA_LJ_MULTI_CHANNEL,
        sample_rate=24000,
        num_channels=2,
        encoding=lib.audio.AudioEncoding.PCM_FLOAT_32_BIT,
        bit_rate="1.54M",
        precision="25-bit",
        num_samples=182015,
    )


def test_get_audio_metadata__bad_file():
    """Test `lib.audio.get_audio_metadata` errors given a none-audio file."""
    with tempfile.NamedTemporaryFile() as file_:
        path = pathlib.Path(file_.name)
        path.write_text("corrupted")
        with pytest.raises(subprocess.CalledProcessError):
            lib.audio.get_audio_metadata(path)


def test_read_audio():
    """Test `lib.audio.read_audio` reads audio that is consistent with it's metadata."""
    metadata = lib.audio.get_audio_metadata(TEST_DATA_LJ)
    audio = lib.audio.read_audio(TEST_DATA_LJ)
    assert audio.dtype == np.float32
    assert audio.shape == (metadata.sample_rate * metadata.length,)


def test_read_audio__slice():
    """Test `lib.audio.read_audio` gets the correct slice."""
    metadata = lib.audio.get_audio_metadata(TEST_DATA_LJ)
    start = 1
    length = 2
    slice_ = lib.audio.read_audio(TEST_DATA_LJ, start, length)
    audio = lib.audio.read_audio(TEST_DATA_LJ)
    expected = audio[start * metadata.sample_rate : (start + length) * metadata.sample_rate]
    np.testing.assert_almost_equal(slice_, expected)


def test_read_wave_audio():
    """Test `lib.audio.read_wave_audio` reads the full audio, correctly."""
    metadata = lib.audio.get_audio_metadata(TEST_DATA_LJ)
    audio = lib.audio.read_wave_audio(metadata)
    np.testing.assert_almost_equal(audio, lib.audio.read_audio(TEST_DATA_LJ))


def test_read_wave_audio__memmap():
    """Test `lib.audio.read_wave_audio` reads the full audio, correctly, into a memory map."""
    metadata = lib.audio.get_audio_metadata(TEST_DATA_LJ)
    audio = lib.audio.read_wave_audio(metadata, memmap=True)
    np.testing.assert_almost_equal(audio, lib.audio.read_audio(TEST_DATA_LJ))


def test_read_wave_audio__slice():
    """Test `lib.audio.read_wave_audio` gets the correct slice."""
    metadata = lib.audio.get_audio_metadata(TEST_DATA_LJ)
    start = 1
    length = 2
    slice_ = lib.audio.read_wave_audio(metadata, start, length)
    audio = lib.audio.read_audio(TEST_DATA_LJ)
    expected = audio[start * metadata.sample_rate : (start + length) * metadata.sample_rate]
    np.testing.assert_almost_equal(slice_, expected)


def test_read_wave_audio__slice__memmap():
    """Test `lib.audio.read_wave_audio` puts the correct slice into a memory map."""
    metadata = lib.audio.get_audio_metadata(TEST_DATA_LJ)
    start = 1
    length = 2
    slice_ = lib.audio.read_wave_audio(metadata, start, length, memmap=True)
    audio = lib.audio.read_audio(TEST_DATA_LJ)
    expected = audio[start * metadata.sample_rate : (start + length) * metadata.sample_rate]
    np.testing.assert_almost_equal(slice_, expected)


def test_read_audio__slice_identity():
    """Test `lib.audio.read_audio` slice is consistent with no slice."""
    metadata = lib.audio.get_audio_metadata(TEST_DATA_LJ)
    audio = lib.audio.read_audio(TEST_DATA_LJ, 0.0, metadata.length)
    np.testing.assert_almost_equal(audio, lib.audio.read_audio(TEST_DATA_LJ))


def test_read_wave_audio__16_bit_pcm():
    """Test `lib.audio.read_wave_audio` handles signed-integer."""
    audio_path = TEST_DATA_PATH / "rate(lj_speech,24000).wav"
    metadata = lib.audio.get_audio_metadata(audio_path)
    start = 1
    length = 2
    slice_ = lib.audio.read_wave_audio(metadata, start, length)
    audio = lib.audio.read_audio(audio_path, dtype=("s16le", "pcm_s16le", np.int16))
    expected = audio[start * metadata.sample_rate : (start + length) * metadata.sample_rate]
    np.testing.assert_almost_equal(slice_, expected)


def test_write_audio__read_audio():
    """Test `lib.audio.write_audio` is consistent with `lib.audio.read_audio` given a 32-bit wav
    file.
    """
    metadata = lib.audio.get_audio_metadata(TEST_DATA_LJ)
    audio = lib.audio.read_audio(TEST_DATA_LJ)
    with tempfile.TemporaryDirectory() as directory:
        copy = pathlib.Path(directory) / "copy.wav"
        lib.audio.write_audio(copy, audio, metadata.sample_rate)
        copy_metadata = lib.audio.get_audio_metadata(copy)
        assert metadata.sample_rate == copy_metadata.sample_rate
        assert metadata.num_channels == copy_metadata.num_channels
        assert metadata.encoding == copy_metadata.encoding
        assert metadata.length == copy_metadata.length
        np.testing.assert_almost_equal(audio, lib.audio.read_audio(copy))


def test_write_audio__overwrite():
    """Test `lib.audio.write_audio` does not overwrite file."""
    with tempfile.NamedTemporaryFile() as file_:
        path = pathlib.Path(file_.name)
        with pytest.raises(ValueError):
            lib.audio.write_audio(path, np.array([0.0], dtype=np.float32), **cf.get())


def test_write_audio__dtype():
    """Test `lib.audio.write_audio` checks dtype."""
    with tempfile.TemporaryDirectory() as directory:
        path = pathlib.Path(directory) / "invalid.wav"
        with pytest.raises(AssertionError):
            audio = np.array([0.0, 0.0, 0.5, -0.5], dtype=np.float64)
            lib.audio.write_audio(path, audio, **cf.get())


def test_write_audio__bounds():
    """Test `lib.audio.write_audio` checks bounds."""
    with tempfile.TemporaryDirectory() as directory:
        path = pathlib.Path(directory) / "invalid.wav"
        with pytest.raises(AssertionError):
            audio = np.array([1.1, 1.2, -3.0, -2.0], dtype=np.float64)
            lib.audio.write_audio(path, audio, **cf.get())


def test_normalize_audio():
    """Test `lib.audio.normalize_audio` normalizes audio."""
    data_type = lib.audio.AudioDataType.SIGNED_INTEGER
    suffix = ".wav"
    bits = 16
    format_ = lib.audio.AudioFormat(
        sample_rate=8000,
        num_channels=2,
        encoding=lib.audio.AudioEncoding.PCM_INT_16_BIT,
        bit_rate="256k",
        precision=f"{bits}-bit",
    )
    temp_dir = tempfile.TemporaryDirectory()
    directory = pathlib.Path(temp_dir.name)
    audio_path = directory / TEST_DATA_LJ.name
    dest_audio_path = directory / ("dest_" + TEST_DATA_LJ.name)
    shutil.copy(TEST_DATA_LJ, audio_path)
    lib.audio.normalize_audio(
        source=audio_path,
        destination=dest_audio_path,
        suffix=suffix,
        data_type=data_type,
        bits=bits,
        sample_rate=format_.sample_rate,
        num_channels=format_.num_channels,
    )
    assert lib.audio.get_audio_metadata(dest_audio_path) == lib.audio.AudioMetadata(
        path=dest_audio_path, num_samples=60672, **lib.utils.dataclass_as_dict(format_)
    )


def test_format_ffmpeg_audio_filter():
    """Test `lib.audio.format_ffmpeg_audio_filter` parameterizes an `ffmpeg` audio
    filter correctly."""
    result = lib.audio.format_ffmpeg_audio_filter(
        "loudnorm", i=-21, lra=4, tp=-6.1, print_format="summary"
    )
    assert result == lib.audio.AudioFilter("loudnorm=i=-21:lra=4:tp=-6.1:print_format=summary")


def test_format_ffmpeg_audio_filters():
    """Test `lib.audio.format_ffmpeg_audio_filters` parameterizes an `ffmpeg` audio
    filter, with multiple filters, correctly."""
    filters = [
        lib.audio.format_ffmpeg_audio_filter(
            "acompressor",
            threshold=0.032,
            ratio=12,
            attack=325,
            release=390,
            knee=6,
            detection="rms",
            makeup=4,
        ),
        lib.audio.format_ffmpeg_audio_filter("equalizer", f=200, t="q", w=0.6, g=-2.4),
        lib.audio.format_ffmpeg_audio_filter(
            "loudnorm", i=-21, lra=4, tp=-6.1, print_format="summary"
        ),
    ]
    result = lib.audio.format_ffmpeg_audio_filters(filters)
    assert result == lib.audio.AudioFilters(
        "acompressor=threshold=0.032:ratio=12:attack=325:release=390:knee=6:detection=rms:makeup=4,"
        "equalizer=f=200:t=q:w=0.6:g=-2.4,loudnorm=i=-21:lra=4:tp=-6.1:print_format=summary"
    )


def test_apply_audio_filters():
    """Test `lib.audio.apply_audio_filters` applies audio filters."""
    loudnorm = lib.audio.format_ffmpeg_audio_filter(
        "loudnorm", i=-21, lra=4, tp=-6.1, print_format="summary"
    )
    audio_filter = lib.audio.format_ffmpeg_audio_filters([loudnorm])
    temp_dir = tempfile.TemporaryDirectory()
    directory = pathlib.Path(temp_dir.name)
    audio_path = directory / TEST_DATA_LJ.name
    dest_audio_path = directory / ("dest_" + TEST_DATA_LJ.name)
    shutil.copy(TEST_DATA_LJ, audio_path)
    metadata = lib.audio.get_audio_metadata(audio_path)
    lib.audio.apply_audio_filters(metadata, dest_audio_path, audio_filter)
    assert lib.audio.get_audio_metadata(dest_audio_path) == lib.audio.AudioMetadata(
        **{**lib.utils.dataclass_as_dict(metadata), **dict(path=dest_audio_path)}
    )


def test_pad_remainder():
    """Test `lib.audio.pad_remainder` pads to a 256 multiple given various edge cases."""
    assert lib.audio.pad_remainder(np.random.normal(size=256), 256).shape[0] == 256
    assert lib.audio.pad_remainder(np.random.normal(size=255), 256).shape[0] == 256
    assert lib.audio.pad_remainder(np.random.normal(size=254), 256).shape[0] == 256


def test_pad_remainder__right():
    """Test `lib.audio.pad_remainder` pads to a 256 multiple on the right side."""
    assert lib.audio.pad_remainder(np.random.normal(size=256), 256, center=False).shape[0] == 256
    assert lib.audio.pad_remainder(np.random.normal(size=255), 256, center=False).shape[0] == 256
    assert lib.audio.pad_remainder(np.random.normal(size=254), 256, center=False).shape[0] == 256


def test_full_scale_sine_wave():
    """Test `lib.audio.full_scale_sine_wave` generates a sine wave."""
    expected = [
        0.0,
        0.4817537,
        0.844328,
        0.9980267,
        0.904827,
        0.58778524,
        0.12533319,
        -0.36812454,
        -0.77051336,
        -0.98228735,
        -0.9510565,
        -0.6845468,
        -0.24868982,
        0.2486897,
        0.6845471,
        0.9510567,
        0.98228717,
        0.7705128,
        0.36812377,
        -0.12533331,
        -0.5877855,
        -0.9048268,
        -0.99802667,
        -0.84432745,
        -0.48175353,
    ]
    np.testing.assert_almost_equal(lib.audio.full_scale_sine_wave(25, 2), np.array(expected))


def test_full_scale_square_wave():
    """Test `lib.audio.full_scale_square_wave` generates a square wave."""
    expected = [
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        -1.0,
        -1.0,
        -1.0,
        -1.0,
        -1.0,
        -1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        -1.0,
        -1.0,
        -1.0,
        -1.0,
        -1.0,
        -1.0,
    ]
    np.testing.assert_almost_equal(lib.audio.full_scale_square_wave(25, 2), np.array(expected))


def test_k_weighting():
    """Test `lib.audio.k_weighting` at several test frequencies."""
    np.testing.assert_almost_equal(
        [-13.9492132, 0.0, 3.3104297],
        lib.audio.k_weighting(np.array([20, lib.audio.REFERENCE_FREQUENCY, 20000]), 24000),
    )


def test_a_weighting():
    """Test `lib.audio.a_weighting` at several test frequencies."""
    np.testing.assert_almost_equal(
        [-50.3856623, 0.0, -9.3315703],
        lib.audio.a_weighting(np.array([20, lib.audio.REFERENCE_FREQUENCY, 20000])),
    )


def test_iso226_weighting():
    """Test `lib.audio.iso226_weighting` based off these charts:
    https://en.wikipedia.org/wiki/A-weighting#/media/File:Lindos3.svg
    https://github.com/wellsaid-labs/Text-to-Speech/blob/adc1f28864c9515a5d33b876b135d2e95da73faf/weights.png
    """
    np.testing.assert_almost_equal(
        [-59.8563292, 0.0, -59.8563292],
        lib.audio.iso226_weighting(np.array([20, lib.audio.REFERENCE_FREQUENCY, 20000])),
    )


def test_identity_weighting():
    """Test `lib.audio.identity_weighting` at several test frequencies."""
    np.testing.assert_almost_equal(
        [0.0, 0.0, 0.0],
        lib.audio.identity_weighting(np.array([20, lib.audio.REFERENCE_FREQUENCY, 20000])),
    )


def test_power_to_amp():
    """Test `amp_to_power` and `lib.audio.power_to_amp` are consistent."""
    input_ = torch.abs(torch.randn(100))
    result = lib.audio.power_to_amp(amp_to_power(input_)).numpy()
    np.testing.assert_almost_equal(result, input_.numpy(), decimal=5)


def test_amp_to_db():
    """Test `lib.audio.amp_to_db` and `lib.audio.db_to_amp` are consistent."""
    input_ = torch.abs(torch.randn(100))
    result = lib.audio.db_to_amp(lib.audio.amp_to_db(input_)).numpy()
    np.testing.assert_almost_equal(result, input_.numpy(), decimal=5)


def test_power_to_db():
    """Test `power_to_db` and `db_to_power` are consistent."""
    input_ = torch.abs(torch.randn(100))
    result = db_to_power(power_to_db(input_)).numpy()
    np.testing.assert_almost_equal(result, input_.numpy(), decimal=5)


def test_power_to_db__numpy():
    """Test `power_to_db` and `db_to_power` are consistent."""
    input_ = torch.abs(torch.randn(100)).numpy()
    result = db_to_power(power_to_db(input_))
    np.testing.assert_almost_equal(result, input_, decimal=5)


def test_power_to_db__float():
    """Test `power_to_db` and `db_to_power` are consistent."""
    input_ = abs(random.random())
    result = db_to_power(power_to_db(input_))
    np.testing.assert_almost_equal(result, input_, decimal=5)


def test_signal_to_rms__full_scale_square_wave():
    """Test `lib.audio.signal_to_rms` on a standard 0 dBFS signal."""
    rms = lib.audio.signal_to_rms(lib.audio.full_scale_square_wave())
    assert rms == pytest.approx(1.0)
    assert lib.audio.amp_to_db(rms) == pytest.approx(0.0)


def test_signal_to_rms__full_scale_sine_wave():
    """Test `lib.audio.signal_to_rms` on a standard -3.01 dBFS signal."""
    rms = lib.audio.signal_to_rms(lib.audio.full_scale_sine_wave())
    assert rms == pytest.approx(0.70710677)
    assert lib.audio.amp_to_db(rms) == pytest.approx(-3.0103001594543457)


def test_signal_to_framed_rms__full_scale_square_wave():
    """Test `lib.audio.signal_to_framed_rms` on a standard 0 dBFS signal."""
    frame_length = 1024
    frame_hop = frame_length // 4
    signal = lib.audio.full_scale_square_wave()
    padded_signal = np.pad(signal, frame_length, mode="constant", constant_values=0)
    frame_rms = lib.audio.signal_to_framed_rms(padded_signal, frame_length, frame_hop)
    assert framed_rms_to_rms(frame_rms, frame_hop, signal.shape[0]) == pytest.approx(1.0)


def test_signal_to_framed_rms__full_scale_sine_wave():
    """Test `lib.audio.signal_to_framed_rms` on a standard -3.01 dBFS signal."""
    frame_length = 1024
    frame_hop = frame_length // 4
    signal = lib.audio.full_scale_sine_wave()
    padded_signal = np.pad(signal, frame_length, mode="constant", constant_values=0)
    frame_rms = lib.audio.signal_to_framed_rms(padded_signal, frame_length, frame_hop)
    assert framed_rms_to_rms(frame_rms, frame_hop, signal.shape[0]) == pytest.approx(0.70710677)


def test_signal_to_framed_rms__signal_to_rms():
    """Test `lib.audio.signal_to_rms` and `lib.audio.signal_to_framed_rms` are equal given a
    appropriately padded signal."""
    for frame_length in range(1, 16):
        iterator = itertools.product(range(1, frame_length + 1), range(1, 32))
        for frame_hop, signal_length in iterator:
            if frame_length % frame_hop != 0:
                continue
            signal = np.random.rand(signal_length)
            # NOTE: Pad the signal such that every sample appears the same number of times
            # (equal to `repeated`). Without  padding the signal, the first sample would appear
            # only once, for example.
            padded_signal = np.pad(signal, frame_length, mode="constant", constant_values=0)
            frame_rms = lib.audio.signal_to_framed_rms(padded_signal, frame_length, frame_hop)
            expected = framed_rms_to_rms(frame_rms, frame_hop, signal.shape[0])
            assert lib.audio.signal_to_rms(signal) == pytest.approx(expected)


def test_power_spectrogram_to_framed_rms__full_scale_square_wave():
    """Test `power_spectrogram_to_framed_rms` on a standard 0 dBFS signal."""
    frame_length = 1024
    frame_hop = frame_length // 4
    window = torch.ones(frame_length)
    signal = lib.audio.full_scale_square_wave()
    padded_signal = np.pad(signal, frame_length, mode="constant", constant_values=0)
    spectrogram = torch.stft(
        torch.tensor(padded_signal),
        n_fft=frame_length,
        hop_length=frame_hop,
        win_length=window.shape[0],
        window=window,
        center=False,
    )
    spectrogram = torch.norm(spectrogram, dim=-1)
    power_spectrogram = amp_to_power(spectrogram).transpose(0, 1)
    frame_rms = power_spectrogram_to_framed_rms(power_spectrogram, window=window).numpy()
    assert framed_rms_to_rms(frame_rms, frame_hop, signal.shape[0]) == pytest.approx(1.0)


def test_power_spectrogram_to_framed_rms__full_scale_sine_wave__sample_rates():
    """Test `power_spectrogram_to_framed_rms` on a standard -3.01 dBFS signal."""
    frame_length = 1024
    frame_hop = frame_length // 4
    window = torch.ones(frame_length)
    signal = lib.audio.full_scale_sine_wave()
    padded_signal = np.pad(signal, frame_length, mode="constant", constant_values=0)
    spectrogram = torch.stft(
        torch.tensor(padded_signal),
        n_fft=frame_length,
        hop_length=frame_hop,
        win_length=window.shape[0],
        window=window,
        center=False,
    )
    spectrogram = torch.norm(spectrogram.double(), dim=-1)
    power_spectrogram = amp_to_power(spectrogram).transpose(0, 1)
    frame_rms = power_spectrogram_to_framed_rms(power_spectrogram, window=window).numpy()
    assert framed_rms_to_rms(frame_rms, frame_hop, signal.shape[0]) == pytest.approx(0.70710677)


def test_power_spectrogram_to_framed_rms__sample_rates():
    """Test `power_spectrogram_to_framed_rms` accross multiple sample rates."""
    for sample_rate in range(1000, 24000, 1000):
        frame_length = 2048
        frame_hop = frame_length // 4
        window = torch.ones(frame_length)
        signal = lib.audio.full_scale_sine_wave(sample_rate)
        padded_signal = np.pad(signal, frame_length, mode="constant", constant_values=0)
        spectrogram = torch.stft(
            torch.tensor(padded_signal),
            n_fft=frame_length,
            hop_length=frame_hop,
            win_length=window.shape[0],
            window=window,
            center=False,
        )
        spectrogram = torch.norm(spectrogram.double(), dim=-1)
        power_spectrogram = amp_to_power(spectrogram).transpose(0, 1)
        frame_rms = power_spectrogram_to_framed_rms(power_spectrogram, window=window).numpy()
        expected = pytest.approx(lib.audio.signal_to_rms(signal))
        assert framed_rms_to_rms(frame_rms, frame_hop, signal.shape[0]) == expected


def test_power_spectrogram_to_framed_rms__window_correction__padding():
    """Test `power_spectrogram_to_framed_rms` window correction using a hann window
    and padding larger than `frame_length`.
    """
    for i in range(1, 5):
        frame_length = 2048
        frame_hop = frame_length // 4
        window = torch.hann_window(frame_length)
        signal = lib.audio.full_scale_sine_wave()
        padded_signal = np.pad(signal, frame_length + i * 100, mode="constant", constant_values=0)
        spectrogram = torch.stft(
            torch.tensor(padded_signal),
            n_fft=frame_length,
            hop_length=frame_hop,
            win_length=window.shape[0],
            window=window,
            center=False,
        )
        spectrogram = torch.norm(spectrogram, dim=-1)
        power_spectrogram = amp_to_power(spectrogram).transpose(0, 1)
        frame_rms = power_spectrogram_to_framed_rms(power_spectrogram, window=window).numpy()
        assert framed_rms_to_rms(frame_rms, frame_hop, signal.shape[0]) == pytest.approx(0.70710677)


def test_power_spectrogram_to_framed_rms__batch():
    """Test `power_spectrogram_to_framed_rms` on a batch of spectrograms."""
    frame_length = 2048
    frame_hop = frame_length // 4
    window = torch.hann_window(frame_length)
    tensors = [
        torch.tensor(lib.audio.full_scale_sine_wave()),
        torch.tensor(lib.audio.full_scale_square_wave()),
    ]
    batched_signal = torch.stack(tensors)
    padded_batched_signal = functional.pad(batched_signal, [frame_length, frame_length])
    batched_spectrogram = torch.stft(
        padded_batched_signal,
        n_fft=frame_length,
        hop_length=frame_hop,
        win_length=frame_length,
        window=window,
        center=False,
    )
    batched_spectrogram = torch.norm(batched_spectrogram, dim=-1)
    batched_power_spectrogram = amp_to_power(batched_spectrogram).transpose(1, 2)
    frame_rms = power_spectrogram_to_framed_rms(batched_power_spectrogram, window=window)
    signal_length = batched_signal.shape[1]
    partial_ = partial(framed_rms_to_rms, frame_hop=frame_hop, signal_length=signal_length)
    assert partial_(frame_rms[0]) == pytest.approx(0.70710677)
    assert partial_(frame_rms[1]) == pytest.approx(1.0)


def test_power_spectrogram_to_framed_rms__zero_elements():
    """Test `power_spectrogram_to_framed_rms` on a zero frames."""
    window = torch.ones(1024)
    power_spectrogram = torch.zeros(64, 0, 1025)
    frame_rms = power_spectrogram_to_framed_rms(power_spectrogram, window=window).numpy()
    assert frame_rms.shape == (64, 0)


def test_signal_to_db_mel_spectrogram():
    """Test `SignalTodBMelSpectrogram` against an equivilant `librosa` implmentation."""
    n_fft = 2048
    win_length = 2048
    hop_length = 512
    amin = 1e-10
    n_mels = 128
    min_decibel = -50.0

    metadata = lib.audio.get_audio_metadata(TEST_DATA_LJ)
    signal = lib.audio.read_audio(TEST_DATA_LJ)

    # Learn more about this algorithm here:
    # https://github.com/librosa/librosa/issues/463#issuecomment-265165830
    S = librosa.stft(
        signal,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        center=False,
        window="hann",
    )
    S = np.abs(S).astype(np.float32) ** 2.0
    log_S = librosa.perceptual_weighting(
        S,
        librosa.fft_frequencies(sr=metadata.sample_rate, n_fft=n_fft),
        amin=amin,
        top_db=None,
    ).astype(np.float32)
    melspec = librosa.feature.melspectrogram(
        S=librosa.db_to_power(log_S),
        sr=metadata.sample_rate,
        n_fft=n_fft,
        htk=True,
        norm=None,
        fmax=None,
        fmin=0.0,
    )
    melspec = librosa.power_to_db(melspec, amin=amin, top_db=None)  # type: ignore
    melspec = np.maximum(melspec, min_decibel).transpose()

    module = cf.partial(SignalTodBMelSpectrogram)(
        fft_length=n_fft,
        frame_hop=hop_length,
        sample_rate=metadata.sample_rate,
        num_mel_bins=n_mels,
        # NOTE: `torch.hann_window` is different than the `scipy` window used by `librosa`.
        # Learn more here: https://github.com/pytorch/audio/issues/452
        window=torch.tensor(librosa.filters.get_window("hann", win_length)).float(),
        min_decibel=min_decibel,
        get_weighting=lambda f, *_: librosa.A_weighting(f),
        lower_hertz=0,
        eps=amin,
    )
    other_mel_spectrogram = module(torch.tensor(signal)).detach().numpy()

    assert melspec.shape == other_mel_spectrogram.shape
    np.testing.assert_allclose(melspec, other_mel_spectrogram, atol=1.6e-3)


def test_signal_to_db_mel_spectrogram__intermediate():
    """Test `SignalTodBMelSpectrogram` intermediate values are consistent:
    - The shapes are correct.
    - The total frame power is preserved between `db_spectrogram` and `db_mel_spectrogram`.
    - The `spectrogram` is correct.
    """
    batch_size = 10
    n_fft = 2048
    hop_length = n_fft // 4
    window = torch.hann_window(n_fft)
    module = cf.partial(SignalTodBMelSpectrogram)(
        fft_length=n_fft,
        frame_hop=hop_length,
        window=window,
        min_decibel=float("-inf"),
        lower_hertz=0,
    )
    tensor = torch.nn.parameter.Parameter(torch.randn(batch_size, 2400))
    db_mel_spectrogram, db_spectrogram, spectrogram = module(tensor, intermediate=True)

    assert spectrogram.shape == (batch_size, db_mel_spectrogram.shape[1], n_fft // 2 + 1)
    assert db_spectrogram.shape == (batch_size, db_mel_spectrogram.shape[1], n_fft // 2 + 1)

    np.testing.assert_allclose(
        db_to_power(db_mel_spectrogram).sum(dim=-1).detach().numpy(),
        db_to_power(db_spectrogram).sum(dim=-1).detach().numpy(),
        rtol=1e-2,
    )

    expected_spectrogram = torch.stft(
        tensor, n_fft, hop_length, win_length=n_fft, window=window, center=False
    )
    expected_spectrogram = torch.norm(expected_spectrogram, dim=-1).transpose(-2, -1)
    np.testing.assert_allclose(
        spectrogram.detach().numpy(), expected_spectrogram.detach().numpy(), rtol=1e-6
    )


def test_signal_to_db_mel_spectrogram__backward():
    """Test `SignalTodBMelSpectrogram` is differentiable."""
    tensor = torch.nn.parameter.Parameter(torch.randn(2400))
    module = SignalTodBMelSpectrogram(**cf.get())
    for output in module(tensor, intermediate=True):
        output.sum().backward(retain_graph=True)


def test_signal_to_db_mel_spectrogram__zeros():
    """Test `SignalTodBMelSpectrogram` is differentiable given zeros."""
    tensor = torch.nn.parameter.Parameter(torch.zeros(2400))
    module = SignalTodBMelSpectrogram(**cf.get())
    for output in module(tensor, intermediate=True):
        output.sum().backward(retain_graph=True)


def test_signal_to_db_mel_spectrogram__batch():
    """Test `SignalTodBMelSpectrogram` is invariant to the batch size."""
    tensor = torch.nn.parameter.Parameter(torch.randn(10, 2400))
    module = SignalTodBMelSpectrogram(**cf.get())
    results = module(tensor)
    result = module(tensor[0])
    _utils.assert_almost_equal(results[0], result, decimal=5)


def test_signal_to_db_mel_spectrogram__padded_batch():
    """Test `SignalTodBMelSpectrogram` is invariant to the batch size and padding."""
    fft_length = 2048
    hop_length = fft_length // 4
    window = torch.from_numpy(librosa.filters.get_window("hann", fft_length)).float()
    batch_size = 10
    signals_ = []
    for i in range(batch_size):
        signal = lib.audio.pad_remainder(torch.randn(2400 + i * 300).numpy(), hop_length)
        signals_.append(torch.from_numpy(signal))
    signals = stack_and_pad_tensors(signals_)
    module = cf.partial(SignalTodBMelSpectrogram)(
        fft_length=fft_length, frame_hop=hop_length, window=window
    )
    expectations = module(signals.tensor, aligned=True)
    for i in range(batch_size):
        result = module(signals_[i], aligned=True)
        expected = expectations[i][: signals.lengths[i] // hop_length]
        _utils.assert_almost_equal(expected, result, decimal=5)


def test_signal_to_db_mel_spectrogram__alignment():
    """Test `SignalTodBMelSpectrogram` can be aligned to the signal."""
    fft_length = 2048
    frame_size = 1200
    frame_hop = frame_size // 4
    metadata = lib.audio.get_audio_metadata(TEST_DATA_LJ)
    signal = lib.audio.read_audio(TEST_DATA_LJ)
    signal = lib.audio.pad_remainder(signal, frame_hop)
    module = cf.partial(SignalTodBMelSpectrogram)(
        fft_length=fft_length,
        frame_hop=frame_hop,
        sample_rate=metadata.sample_rate,
        window=torch.hann_window(frame_size),
    )
    db_mel_spectrogram = module(torch.tensor(signal), aligned=True).numpy()
    assert db_mel_spectrogram.dtype == np.float32
    assert len(db_mel_spectrogram.shape) == 2
    assert int(signal.shape[0]) / int(db_mel_spectrogram.shape[0]) == frame_hop


def _db_spectrogram_to_loudness(db_spectrogram: torch.Tensor, window: torch.Tensor) -> float:
    """Get loudness as defined by ITU-R BS.1770-4 from a k-weighted dB spectrogram."""
    power_spectrogram = db_to_power(db_spectrogram)
    loudness = power_spectrogram_to_framed_rms(power_spectrogram, window=window)
    loudness = torch.tensor([v for v in loudness if lib.audio.amp_to_db(v) >= -70])
    if len(loudness) == 0:
        return -70.0
    relative_gate = power_to_db(amp_to_power(loudness).mean()) - 10
    loudness = torch.tensor([v for v in loudness if lib.audio.amp_to_db(v) >= relative_gate])
    return typing.cast(float, power_to_db(amp_to_power(loudness).mean()).item())


def test__loudness():
    """Test that `lib.audio` can be used to measure loudness as defined by ITU-R BS.1770-4.

    TODO: Test against the original DeMan algorithm: https://github.com/BrechtDeMan/loudness.py
    """
    for path in TEST_DATA_PATH.glob("*.wav"):
        metadata = lib.audio.get_audio_metadata(path)
        signal = lib.audio.read_audio(path)

        # NOTE: The original implementation doesn't pad the signal at all; therefore, the boundary
        # samples are sampled less frequency than other samples.
        fft_length = int(metadata.sample_rate * 0.4)
        window = torch.ones(fft_length)
        signal_to_db_spectrogram = SignalTodBMelSpectrogram(
            fft_length=fft_length,
            frame_hop=fft_length // 4,
            sample_rate=metadata.sample_rate,
            num_mel_bins=128,
            window=window,
            min_decibel=float("-inf"),
            # NOTE: Our `k_weighting` implementation predicts a different `offset` than -0.691 which
            # is required by the original guidelines.
            get_weighting=partial(lib.audio.k_weighting, offset=-0.691),
            eps=1e-10,
            lower_hertz=0,
            upper_hertz=20000,
        )

        # NOTE: Our K-Weighting implementation is also based off DeMan's work.
        meter = lib.audio.get_pyloudnorm_meter(metadata.sample_rate, "DeMan")

        signal = lib.audio.pad_remainder(signal, fft_length // 4, False)
        db_mel_spectrogram, db_spectrogram, spectrogram = signal_to_db_spectrogram(
            torch.tensor(signal), intermediate=True
        )

        # Test mel dB spectrogram
        loudness = _db_spectrogram_to_loudness(db_mel_spectrogram, window)
        assert abs(loudness - meter.integrated_loudness(signal)) < 0.004

        # Test regular dB spectrogram
        loudness = _db_spectrogram_to_loudness(db_spectrogram, window)
        assert abs(loudness - meter.integrated_loudness(signal)) < 0.004


def test_griffin_lim():
    """Test that `lib.audio.griffin_lim` executes."""
    metadata = lib.audio.get_audio_metadata(TEST_DATA_LJ)
    signal = lib.audio.read_audio(TEST_DATA_LJ)
    signal = lib.audio.pad_remainder(signal, **cf.get())
    module = cf.partial(SignalTodBMelSpectrogram)(sample_rate=metadata.sample_rate)
    db_mel_spectrogram = module(torch.tensor(signal), aligned=True).numpy()
    waveform = cf.partial(lib.audio.griffin_lim)(
        db_mel_spectrogram, sample_rate=metadata.sample_rate
    )
    assert waveform.shape[0] == signal.shape[0]


def test_griffin_lim__large_numbers():
    """Test that `lib.audio.griffin_lim` produces empty array for large numbers."""
    audio = np.random.uniform(low=-2000, high=2000, size=(50, 50))
    shape = lib.audio.griffin_lim(audio, **cf.get()).shape
    assert shape == (0,)


def test_griffin_lim__small_array():
    """Test that `lib.audio.griffin_lim` produces empty array for a small array."""
    audio = np.random.uniform(low=-1, high=1, size=(1, 1))
    audio = lib.audio.griffin_lim(audio, **cf.get())
    assert audio.shape == (0,)


def test_highpass_filter():
    """Test that `lib.audio.highpass_filter` creates an array of the right shape."""
    sample_rate = 8000
    signal = np.random.uniform(low=-1, high=1, size=(sample_rate,))
    filtered = lib.audio.highpass_filter(signal, 300, sample_rate=sample_rate)
    assert filtered.shape == signal.shape


def test_get_non_speech_segments():
    """Test that `lib.audio.get_non_speech_segments` gets the correct non speech segments in audio
    file."""
    low_cut = 300
    frame_length = 50
    hop_length = 5
    threshold = -60
    metadata = lib.audio.get_audio_metadata(TEST_DATA_LJ)
    audio = lib.audio.read_audio(TEST_DATA_LJ)
    segments = lib.audio.get_non_speech_segments(
        audio, metadata, low_cut, frame_length, hop_length, threshold
    )
    assert segments == [
        (0, 0.009958333333333333),
        (2.75, 2.9849583333333336),
        (4.885, 5.289958333333334),
        (7.38, 7.569958333333333),
        (7.55, 7.583958333333333),
    ]

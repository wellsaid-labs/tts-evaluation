from unittest import mock

import pathlib

from torchnlp.text_encoders import CharacterEncoder

import numpy

from src.datasets.process import _predict_spectrogram
from src.datasets.process import compute_spectrogram
from src.datasets.process import normalize_audio
from src.datasets.process import process_with_processes
from src.datasets.process import split_dataset
from src.spectrogram_model import SpectrogramModel
from src.utils import Checkpoint


@mock.patch("src.datasets.process.os.system")
def test_normalize_audio(mock_os_system):
    mock_os_system.return_value = None

    audio_path = pathlib.Path('tests/_test_data/lj_speech.wav')
    normalized_audio_path = normalize_audio(
        audio_path,
        resample=24000,
        norm=True,
        guard=True,
        lower_hertz=20,
        upper_hertz=12000,
        loudness=True)

    assert 'rate' in normalized_audio_path.stem
    assert '24000' in normalized_audio_path.stem
    assert 'norm' in normalized_audio_path.stem
    assert 'loudness' in normalized_audio_path.stem
    assert 'guard' in normalized_audio_path.stem
    assert 'sinc' in normalized_audio_path.stem
    assert '20,12000' in normalized_audio_path.stem


@mock.patch("src.datasets.process.os.system")
def test_normalize_audio_not_normalized(mock_os_system):
    mock_os_system.return_value = None

    audio_path = pathlib.Path('tests/_test_data/lj_speech.wav')
    normalized_audio_path = normalize_audio(
        audio_path,
        resample=None,
        norm=False,
        guard=False,
        lower_hertz=None,
        upper_hertz=None,
        loudness=None)

    assert not mock_os_system.called
    assert normalized_audio_path == audio_path


@mock.patch("src.datasets.process.numpy.save")
def test_compute_spectrogram(mock_save):
    text_encoder = CharacterEncoder(['this is a test'])
    text = 'this is a test'
    checkpoint = Checkpoint(
        directory='run/09_10/norm',
        model=SpectrogramModel(text_encoder.vocab_size),
        step=0,
        text_encoder=text_encoder)
    checkpoint.path = pathlib.Path('run/09_10/norm/step_123.pt')
    audio_path = pathlib.Path('tests/_test_data/lj_speech_24000.wav')
    expected_dest_spectrogram = 'tests/_test_data/spectrogram(lj_speech_24000).npy'
    expected_dest_padded_audio = 'tests/_test_data/pad(lj_speech_24000).npy'
    expected_dest_predicted_spectrogram = ('tests/_test_data/predicted_spectrogram'
                                           '(lj_speech_24000,run_09_10_norm_step_123_pt).npy')
    mock_save.return_value = None
    dest_padded_audio, dest_spectrogram, dest_predicted_spectrogram = compute_spectrogram(
        audio_path, text, checkpoint)
    assert pathlib.Path(expected_dest_spectrogram) == dest_spectrogram
    assert pathlib.Path(expected_dest_padded_audio) == dest_padded_audio
    assert pathlib.Path(expected_dest_predicted_spectrogram) == dest_predicted_spectrogram


def test__predict_spectrogram():
    text_encoder = CharacterEncoder(['this is a test'])
    frame_channels = 80
    num_frames = 10
    checkpoint = Checkpoint(
        directory='.',
        model=SpectrogramModel(text_encoder.vocab_size, frame_channels=frame_channels),
        step=0,
        text_encoder=text_encoder)
    real_spectrogram = numpy.zeros((num_frames, frame_channels), dtype=numpy.float32)
    predicted_spectrogram = _predict_spectrogram(checkpoint, 'this is a test', real_spectrogram)
    assert predicted_spectrogram.shape == (num_frames, frame_channels)


def test_split_dataset():
    dataset = [1, 2, 3, 4, 5]
    splits = (.6, .2, .2)
    assert split_dataset(dataset, splits, random_seed=None) == [[1, 2, 3], [4], [5]]


def test_split_dataset_shuffle():
    dataset = [1, 2, 3, 4, 5]
    splits = (.6, .2, .2)
    assert split_dataset(dataset, splits, random_seed=123) == [[4, 2, 5], [3], [1]]


def test_split_dataset_rounding():
    dataset = [1]
    splits = (.33, .33, .34)
    assert split_dataset(dataset, splits, random_seed=123) == [[], [], [1]]


def mock_func(a):
    return a**2


def test_process_with_processes():
    expected = [1, 4, 9]

    processed = process_with_processes([1, 2, 3], mock_func)
    assert expected == processed

from unittest import mock

import pathlib

from torchnlp.text_encoders import CharacterEncoder
from torchnlp.text_encoders import IdentityEncoder

import numpy
import torch

from src.datasets import Speaker
from src.datasets import SpectrogramTextSpeechRow
from src.datasets import TextSpeechRow
from src.datasets.process import _compute_spectrogram
from src.datasets.process import _normalize_audio_and_cache
from src.datasets.process import _predict_spectrogram
from src.datasets.process import _process_in_parallel
from src.datasets.process import _split_dataset
from src.datasets.process import balance_dataset
from src.datasets.process import compute_spectrograms
from src.spectrogram_model import SpectrogramModel
from src.utils import Checkpoint
from src.utils import ROOT_PATH


def test_balance_dataset():
    balanced = balance_dataset(['a', 'a', 'b', 'b', 'c'], lambda x: x)
    assert len(balanced) == 3
    assert len(set(balanced)) == 3


@mock.patch('src.datasets.process._process_in_parallel', return_value=[])
@mock.patch('src.datasets.process._predict_spectrogram', return_value=[])
def test_compute_spectrograms(_, __):
    assert compute_spectrograms([], True, '', batch_size=5, device=torch.device('cpu')) == []


@mock.patch('src.datasets.process.os.system')
def test__normalize_audio_and_cache(mock_os_system):
    mock_os_system.return_value = None

    audio_path = pathlib.Path('tests/_test_data/lj_speech.wav')
    normalized_audio_path = _normalize_audio_and_cache(
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


@mock.patch('src.datasets.process.os.system')
def test__normalize_audio_and_cache_not_normalized(mock_os_system):
    mock_os_system.return_value = None

    audio_path = pathlib.Path('tests/_test_data/lj_speech.wav')
    normalized_audio_path = _normalize_audio_and_cache(
        audio_path,
        resample=None,
        norm=False,
        guard=False,
        lower_hertz=None,
        upper_hertz=None,
        loudness=None)

    assert not mock_os_system.called
    assert normalized_audio_path == audio_path


@mock.patch('src.datasets.process.numpy.save')
@mock.patch('src.datasets.process.numpy.load')
def test__compute_spectrogram_not_on_disk(mock_load, mock_save):
    text = 'this is a test'
    audio_path = pathlib.Path('tests/_test_data/lj_speech_24000.wav')
    row = TextSpeechRow(
        text=text, speaker=Speaker.LINDA_JOHNSON, audio_path=audio_path, metadata=None)
    expected_dest_spectrogram = 'tests/_test_data/spectrogram(lj_speech_24000).npy'
    expected_dest_padded_audio = 'tests/_test_data/pad(lj_speech_24000).npy'
    mock_save.return_value = None
    mock_load.return_value = numpy.array([1])
    spectrogram_row = _compute_spectrogram(row, on_disk=False)
    assert torch.is_tensor(spectrogram_row.spectrogram)
    assert torch.is_tensor(spectrogram_row.spectrogram_audio)

    spectrogram_row = _compute_spectrogram(row, on_disk=True)
    assert pathlib.Path(expected_dest_spectrogram) == spectrogram_row.spectrogram.path
    assert pathlib.Path(expected_dest_padded_audio) == spectrogram_row.spectrogram_audio.path


@mock.patch('src.datasets.process.numpy.save')
@mock.patch('src.datasets.process.numpy.load')
@mock.patch('src.datasets.process.Checkpoint.from_path')
def test__predict_spectrogram(mock_from_path, mock_load, mock_save):
    text_encoder = CharacterEncoder(['this is a test'])
    speaker_encoder = IdentityEncoder([Speaker.LINDA_JOHNSON])
    frame_channels = 80
    num_frames = 10
    audio_path = pathlib.Path('tests/_test_data/lj_speech_24000.wav')
    expected_path_predicted_spectrogram = (
        'tests/_test_data/predicted_spectrogram'
        '(lj_speech_24000,run_09_10_norm_step_123_pt,aligned=True).npy')
    mock_from_path.return_value = Checkpoint(
        path=ROOT_PATH / 'run/09_10/norm/step_123.pt',
        directory='.',
        model=SpectrogramModel(
            text_encoder.vocab_size, speaker_encoder.vocab_size, frame_channels=frame_channels),
        step=0,
        speaker_encoder=speaker_encoder,
        text_encoder=text_encoder)
    row = SpectrogramTextSpeechRow(
        text='this is a test',
        audio_path=audio_path,
        speaker=Speaker.LINDA_JOHNSON,
        spectrogram=torch.rand(num_frames, frame_channels),
        spectrogram_audio=torch.rand(num_frames * 10),
        predicted_spectrogram=None,
        metadata=None)
    mock_save.return_value = None
    mock_load.return_value = numpy.array([1])
    rows = _predict_spectrogram([row], '', torch.device('cpu'), 1, on_disk=False)
    assert len(rows) == 1
    assert torch.is_tensor(rows[0].predicted_spectrogram)

    rows = _predict_spectrogram([row], '', torch.device('cpu'), 1, on_disk=True)
    assert len(rows) == 1
    assert str(rows[0].predicted_spectrogram.path) == expected_path_predicted_spectrogram


def test__split_dataset():
    dataset = [1, 2, 3, 4, 5]
    splits = (.6, .2, .2)
    assert _split_dataset(dataset, splits, random_seed=None) == [[1, 2, 3], [4], [5]]


def test__split_dataset_shuffle():
    dataset = [1, 2, 3, 4, 5]
    splits = (.6, .2, .2)
    assert _split_dataset(dataset, splits, random_seed=123) == [[4, 2, 5], [3], [1]]


def test__split_dataset_rounding():
    dataset = [1]
    splits = (.33, .33, .34)
    assert _split_dataset(dataset, splits, random_seed=123) == [[], [], [1]]


def mock_func(a):
    return a**2


def test__process_in_parallel():
    expected = [1, 4, 9]

    processed = _process_in_parallel([1, 2, 3], mock_func)
    assert expected == processed

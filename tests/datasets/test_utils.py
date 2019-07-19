from unittest import mock

import pathlib

import numpy
import torch

from src.datasets import Gender
from src.datasets import Speaker
from src.datasets import TextSpeechRow
from src.datasets.utils import add_predicted_spectrogram_column
from src.datasets.utils import add_spectrogram_column
from src.datasets.utils import filter_
from src.environment import ROOT_PATH
from src.environment import TTS_DISK_CACHE_NAME
from src.spectrogram_model import InputEncoder
from src.spectrogram_model import SpectrogramModel
from src.utils import Checkpoint


def test_filter_():
    a = TextSpeechRow(text='this is a test', speaker=Speaker('Stay', Gender.FEMALE), audio_path='')
    b = TextSpeechRow(text='this is a test', speaker=Speaker('Stay', Gender.FEMALE), audio_path='')
    c = TextSpeechRow(
        text='this is a test', speaker=Speaker('Remove', Gender.FEMALE), audio_path='')

    assert filter_(lambda e: e.speaker != Speaker('Remove', Gender.FEMALE), [a, b, c]) == [a, b]


@mock.patch('src.utils.on_disk_tensor.np.save')
@mock.patch('src.utils.on_disk_tensor.np.load')
@mock.patch('src.datasets.utils.Checkpoint.from_path')
def test_add_predicted_spectrogram_column(mock_from_path, mock_load, mock_save):
    speaker = Speaker('Test Speaker', Gender.MALE)
    input_encoder = InputEncoder(['this is a test'], [speaker])
    frame_channels = 124
    num_frames = 10
    audio_path = pathlib.Path('tests/_test_data/lj_speech_24000.wav')
    checkpoint = Checkpoint(
        directory='.',
        model=SpectrogramModel(
            input_encoder.text_encoder.vocab_size,
            input_encoder.speaker_encoder.vocab_size,
            frame_channels=frame_channels),
        step=0,
        input_encoder=input_encoder)
    checkpoint.path = ROOT_PATH / 'run/09_10/norm/step_123.pt'
    mock_from_path.return_value = checkpoint
    row = TextSpeechRow(
        text='this is a test',
        audio_path=audio_path,
        speaker=speaker,
        spectrogram=torch.rand(num_frames, frame_channels))
    rows = [row]
    mock_save.return_value = None
    mock_load.return_value = numpy.array([1])

    # In memory
    rows = add_predicted_spectrogram_column(rows, '', torch.device('cpu'), 1, on_disk=False)
    assert len(rows) == 1
    assert torch.is_tensor(rows[0].predicted_spectrogram)

    # On disk
    expected_path_predicted_spectrogram = (
        'tests/_test_data/' + TTS_DISK_CACHE_NAME + '/predicted_spectrogram'
        '(lj_speech_24000,run_09_10_norm_step_123_pt,aligned=True).npy')
    rows = add_predicted_spectrogram_column(rows, '', torch.device('cpu'), 1, on_disk=True)
    assert len(rows) == 1
    assert str(rows[0].predicted_spectrogram.path) == expected_path_predicted_spectrogram

    # On disk and cached
    with mock.patch('src.datasets.utils.pathlib.Path.is_file') as mock_is_file:
        mock_is_file.return_value = True
        rows = add_predicted_spectrogram_column(rows, '', torch.device('cpu'), 1, on_disk=True)
        assert len(rows) == 1
        assert str(rows[0].predicted_spectrogram.path) == expected_path_predicted_spectrogram

    # No audio path
    rows = [row._replace(audio_path=None)]
    expected_path_predicted_spectrogram = (
        '/tmp/' + TTS_DISK_CACHE_NAME + '/predicted_spectrogram(lj_speech_24000,')
    rows = add_predicted_spectrogram_column(rows, '', torch.device('cpu'), 1, on_disk=True)
    assert len(rows) == 1
    assert '/tmp/' + TTS_DISK_CACHE_NAME + '/predicted_spectrogram(' in str(
        rows[0].predicted_spectrogram.path)
    assert ',run_09_10_norm_step_123_pt,aligned=True).npy' in str(
        rows[0].predicted_spectrogram.path)
    assert 'lj_speech_24000' not in str(rows[0].predicted_spectrogram.path)


@mock.patch('src.utils.on_disk_tensor.np.save')
@mock.patch('src.utils.on_disk_tensor.np.load')
def test_add_spectrogram_column(mock_load, mock_save):
    audio_path = pathlib.Path('tests/_test_data/lj_speech_24000.wav')
    rows = [
        TextSpeechRow(
            text='this is a test',
            speaker=Speaker('Test Speaker', Gender.MALE),
            audio_path=audio_path,
            metadata=None)
    ]
    mock_save.return_value = None
    mock_load.return_value = numpy.array([1])

    # In memory
    processed = add_spectrogram_column(rows, on_disk=False)
    assert torch.is_tensor(processed[0].spectrogram)
    assert torch.is_tensor(processed[0].spectrogram_audio)

    # On disk
    expected_dest_spectrogram = (
        'tests/_test_data/' + TTS_DISK_CACHE_NAME + '/spectrogram(lj_speech_24000).npy')
    expected_dest_padded_audio = (
        'tests/_test_data/' + TTS_DISK_CACHE_NAME + '/pad(lj_speech_24000).npy')
    processed = add_spectrogram_column(rows, on_disk=True)
    assert pathlib.Path(expected_dest_spectrogram) == processed[0].spectrogram.path
    assert pathlib.Path(expected_dest_padded_audio) == processed[0].spectrogram_audio.path

    # On disk and cached
    with mock.patch('src.datasets.utils.pathlib.Path.is_file') as mock_is_file:
        mock_is_file.return_value = True
        processed = add_spectrogram_column(rows, on_disk=True)
        assert pathlib.Path(expected_dest_spectrogram) == processed[0].spectrogram.path
        assert pathlib.Path(expected_dest_padded_audio) == processed[0].spectrogram_audio.path

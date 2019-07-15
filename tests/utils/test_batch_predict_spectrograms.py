from unittest import mock

import numpy
import torch

from src.datasets import Gender
from src.datasets import Speaker
from src.datasets import TextSpeechRow
from src.spectrogram_model import InputEncoder
from src.spectrogram_model import SpectrogramModel
from src.utils.batch_predict_spectrograms import batch_predict_spectrograms


@mock.patch('src.utils.on_disk_tensor.np.save')
@mock.patch('src.utils.on_disk_tensor.np.load')
def test_batch_predict_spectrograms(mock_load, mock_save):
    speaker = Speaker('Test Speaker', Gender.FEMALE)
    input_encoder = InputEncoder(['this is a test'], [speaker])
    frame_channels = 128
    model = SpectrogramModel(
        input_encoder.text_encoder.vocab_size,
        input_encoder.speaker_encoder.vocab_size,
        frame_channels=frame_channels)
    data = [TextSpeechRow(text='this is a test', audio_path=None, speaker=speaker)]
    mock_save.return_value = None
    mock_load.return_value = numpy.array([1])

    predictions = batch_predict_spectrograms(
        data=data,
        input_encoder=input_encoder,
        model=model,
        batch_size=1,
        device=torch.device('cpu'),
        aligned=False)
    assert len(predictions) == 1
    assert torch.is_tensor(predictions[0])

    predictions = batch_predict_spectrograms(
        data=data,
        input_encoder=input_encoder,
        model=model,
        batch_size=1,
        device=torch.device('cpu'),
        filenames=['/tmp/tensor.npy'],
        aligned=False)
    assert len(predictions) == 1
    assert '/tmp/tensor.npy' in str(predictions[0].path)


@mock.patch('src.utils.on_disk_tensor.np.save')
@mock.patch('src.utils.on_disk_tensor.np.load')
def test_batch_predict_spectrograms_sorting(mock_load, mock_save):
    speaker = Speaker('Test Speaker', Gender.FEMALE)
    input_encoder = InputEncoder(['this is a test'], [speaker])
    frame_channels = 128
    model = SpectrogramModel(
        input_encoder.text_encoder.vocab_size,
        input_encoder.speaker_encoder.vocab_size,
        frame_channels=frame_channels)
    data = [TextSpeechRow(text='this is a test', audio_path=None, speaker=speaker)] * 20
    filenames = ['/tmp/tensor_%d.npy' % d for d in range(20)]
    mock_save.return_value = None
    mock_load.return_value = numpy.array([1])

    predictions = batch_predict_spectrograms(
        data=data,
        input_encoder=input_encoder,
        model=model,
        batch_size=1,
        device=torch.device('cpu'),
        filenames=filenames,
        aligned=False)
    assert len(filenames) == 20
    # Ensure predictions are sorted in the right order
    for prediction, filename in zip(predictions, filenames):
        assert filename in str(prediction.path)


@mock.patch('src.utils.on_disk_tensor.np.save')
@mock.patch('src.utils.on_disk_tensor.np.load')
def test_batch_predict_spectrograms__aligned(mock_load, mock_save):
    speaker = Speaker('Test Speaker', Gender.FEMALE)
    input_encoder = InputEncoder(['this is a test'], [speaker])
    frame_channels = 128
    model = SpectrogramModel(
        input_encoder.text_encoder.vocab_size,
        input_encoder.speaker_encoder.vocab_size,
        frame_channels=frame_channels)
    spectrogram = torch.rand(10, frame_channels)
    example = TextSpeechRow(
        text='this is a test', audio_path=None, speaker=speaker, spectrogram=spectrogram)
    mock_save.return_value = None
    mock_load.return_value = numpy.array([1])

    predictions = batch_predict_spectrograms(
        data=[example],
        input_encoder=input_encoder,
        model=model,
        batch_size=1,
        device=torch.device('cpu'),
        aligned=True)
    assert len(predictions) == 1
    assert torch.is_tensor(predictions[0])

from unittest import mock

from torchnlp.text_encoders import CharacterEncoder
from torchnlp.text_encoders import IdentityEncoder

import torch
import numpy

from src.bin.train.spectrogram_model.data_iterator import DataBatchIterator
from src.bin.train.spectrogram_model.data_iterator import DataLoader
from src.datasets import Speaker


class MockDataset():

    def __init__(self, rows, spectrogram_lengths):
        self.rows = rows
        self.spectrogram_lengths = spectrogram_lengths

    def __getitem__(self, index):
        return self.rows[index]


@mock.patch('src.bin.train.spectrogram_model.data_iterator.np.load')
def test_data_loader(mock_load):
    text_key = 'text'
    speaker_key = 'speaker'
    spectrogram_path_key = 'spectrogram_path'
    data = [{
        spectrogram_path_key: 'spectrogram_path',
        text_key: 'text',
        speaker_key: Speaker.LINDA_JOHNSON,
    }]
    spectrogram_length = 10
    spectrogram = numpy.zeros((spectrogram_length, 80), dtype=numpy.float32)
    mock_load.return_value = spectrogram
    text_encoder = CharacterEncoder(['text'])
    speaker_encoder = IdentityEncoder([Speaker.LINDA_JOHNSON])
    data_loader = DataLoader(
        data,
        text_encoder,
        speaker_encoder,
        text_key=text_key,
        spectrogram_path_key=spectrogram_path_key)
    assert data_loader.spectrogram_lengths == [spectrogram_length]
    assert len(data_loader) == 1
    assert set(data_loader[0].keys()) == set(['text', 'spectrogram', 'stop_token', 'speaker'])


@mock.patch('src.bin.train.spectrogram_model.data_iterator.DataLoader')
def test_data_iterator(MockDataLoader):
    MockDataLoader.return_value = MockDataset(
        rows=[{
            'text': torch.LongTensor([1, 2, 3]),
            'spectrogram': torch.FloatTensor(500, 80),
            'stop_token': torch.FloatTensor(500),
            'speaker': torch.LongTensor([1]),
        }, {
            'text': torch.LongTensor([1, 2]),
            'spectrogram': torch.FloatTensor(450, 80),
            'stop_token': torch.FloatTensor(450),
            'speaker': torch.LongTensor([1]),
        }],
        spectrogram_lengths=[500, 450])
    batch_size = 1
    text_encoder = None
    speaker_encoder = None
    data = []

    iterator = DataBatchIterator(data, text_encoder, speaker_encoder, batch_size,
                                 torch.device('cpu'))
    assert len(iterator) == 2
    next(iter(iterator))

    iterator = DataBatchIterator(
        data, text_encoder, speaker_encoder, batch_size, torch.device('cpu'), trial_run=True)
    assert len(iterator) == 1
    iterator = iter(iterator)
    next(iterator)
    try:
        next(iterator)
    except StopIteration:
        error = True
    assert error

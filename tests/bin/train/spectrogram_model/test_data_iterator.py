from unittest import mock

from torchnlp.text_encoders import CharacterEncoder

import torch
import numpy

from src.bin.train.spectrogram_model.data_iterator import DataBatchIterator
from src.bin.train.spectrogram_model.data_iterator import DataLoader


class MockDataset():

    def __init__(self, rows, spectrogram_lengths):
        self.rows = rows
        self.spectrogram_lengths = spectrogram_lengths

    def __getitem__(self, index):
        return self.rows[index]


@mock.patch('src.bin.train.spectrogram_model.data_iterator.np.load')
def test_data_loader(mock_load):
    text_key = 'text'
    spectrogram_path_key = 'spectrogram_path'
    other_keys = ['speaker']
    data = [{spectrogram_path_key: 'spectrogram_path', text_key: 'text', other_keys[0]: 'speaker'}]
    spectrogram_length = 10
    spectrogram = numpy.zeros((spectrogram_length, 80), dtype=numpy.float32)
    mock_load.return_value = spectrogram
    text_encoder = CharacterEncoder(['text'])
    data_loader = DataLoader(
        data,
        text_encoder,
        text_key=text_key,
        spectrogram_path_key=spectrogram_path_key,
        other_keys=other_keys)
    assert data_loader.spectrogram_lengths == [spectrogram_length]
    assert len(data_loader) == 1
    assert set(data_loader[0].keys()) == set(['text', 'spectrogram', 'stop_token'] + other_keys)


@mock.patch('src.bin.train.spectrogram_model.data_iterator.DataLoader')
def test_data_iterator(MockDataLoader):
    MockDataLoader.return_value = MockDataset(
        rows=[{
            'text': torch.LongTensor([1, 2, 3]),
            'spectrogram': torch.FloatTensor(500, 80),
            'stop_token': torch.FloatTensor(500),
        }, {
            'text': torch.LongTensor([1, 2]),
            'spectrogram': torch.FloatTensor(450, 80),
            'stop_token': torch.FloatTensor(450),
        }],
        spectrogram_lengths=[500, 450])
    batch_size = 1
    text_encoder = None
    data = []

    iterator = DataBatchIterator(data, text_encoder, batch_size, torch.device('cpu'))
    assert len(iterator) == 2
    next(iter(iterator))

    iterator = DataBatchIterator(
        data, text_encoder, batch_size, torch.device('cpu'), trial_run=True)
    assert len(iterator) == 1
    iterator = iter(iterator)
    next(iterator)
    try:
        next(iterator)
    except StopIteration:
        error = True
    assert error

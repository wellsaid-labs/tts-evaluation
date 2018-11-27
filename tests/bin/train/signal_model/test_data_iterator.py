from unittest import mock

import torch
import numpy

from src.bin.train.signal_model.data_iterator import DataBatchIterator
from src.bin.train.signal_model.data_iterator import DataLoader


@mock.patch('src.bin.train.spectrogram_model.data_iterator.np.load')
@mock.patch('src.bin.train.signal_model.data_iterator.random.randint')
def test_data_loader(randint_mock, mock_load):
    audio_path_key = 'aligned_audio_path'
    spectrogram_path_key = 'predicted_spectrogram_path'
    randint_mock.return_value = 5
    samples_per_frame = 10
    spectrogram_channels = 80
    spectrogram = numpy.random.rand(10, spectrogram_channels)
    signal = numpy.random.rand(100)
    mock_load.side_effect = [signal, spectrogram]
    slice_pad = 3
    slice_size = 3
    data_loader = DataLoader(
        [{
            audio_path_key: 'audio.npy',
            spectrogram_path_key: 'spectrogram.npy',
        }],
        audio_path_key=audio_path_key,
        spectrogram_path_key=spectrogram_path_key,
        slice_size=slice_size,
        slice_pad=slice_pad)
    assert len(data_loader) == 1
    example = data_loader[0]

    assert example['slice']['spectrogram'].shape == (slice_size + slice_pad * 2,
                                                     spectrogram_channels)
    assert example['slice']['input_signal'].shape == (slice_size * samples_per_frame, 2)
    assert example['slice']['target_signal_coarse'].shape == (slice_size * samples_per_frame,)
    assert example['slice']['target_signal_fine'].shape == (slice_size * samples_per_frame,)


@mock.patch('src.bin.train.signal_model.data_iterator.DataLoader')
def test_data_batch_iterator(MockDataLoader):
    MockDataLoader.return_value = [{
        'slice': {
            'input_signal': torch.FloatTensor(100, 2),
            'target_signal_coarse': torch.FloatTensor(100,),
            'target_signal_fine': torch.FloatTensor(100,),
            'spectrogram': torch.FloatTensor(50, 80),
        },
        'spectrogram': torch.FloatTensor(50, 80),
        'signal': torch.FloatTensor(100),
    }, {
        'slice': {
            'input_signal': torch.FloatTensor(90, 2),
            'target_signal_coarse': torch.FloatTensor(90,),
            'target_signal_fine': torch.FloatTensor(90,),
            'spectrogram': torch.FloatTensor(50, 80),
        },
        'spectrogram': torch.FloatTensor(50, 80),
        'signal': torch.FloatTensor(90),
    }]
    device = torch.device('cpu')
    batch_size = 2
    iterator = DataBatchIterator([], batch_size, device)
    assert len(iterator) == 1
    item = next(iter(iterator))

    assert torch.sum(item['slice']['signal_mask']) == 190

    iterator = DataBatchIterator([], batch_size, device, trial_run=True)
    assert len(iterator) == 1
    iterator = iter(iterator)
    next(iterator)
    try:
        next(iterator)
    except StopIteration:
        error = True
    assert error

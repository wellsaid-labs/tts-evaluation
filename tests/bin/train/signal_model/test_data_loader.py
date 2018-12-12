from unittest import mock

import torch

from src.bin.train.signal_model.data_loader import _get_slice
from src.bin.train.signal_model.data_loader import DataLoader

from tests.bin.train.utils import get_example_spectrogram_text_speech_rows


@mock.patch('src.bin.train.signal_model.data_loader.random.randint')
def test__get_slice(randint_mock):
    randint_mock.return_value = 5
    samples_per_frame = 10
    spectrogram_channels = 80
    spectrogram = torch.rand(10, spectrogram_channels)
    signal = torch.rand(100)
    slice_pad = 3
    slice_size = 3
    slice_ = _get_slice(spectrogram, signal, slice_size=slice_size, slice_pad=slice_pad)

    assert slice_.input_spectrogram.shape == (slice_size + slice_pad * 2, spectrogram_channels)
    assert slice_.input_signal.shape == (slice_size * samples_per_frame, 2)
    assert slice_.target_signal_coarse.shape == (slice_size * samples_per_frame,)
    assert slice_.target_signal_fine.shape == (slice_size * samples_per_frame,)


def test_data_loader():
    samples_per_frame = 2
    data = get_example_spectrogram_text_speech_rows(samples_per_frame=samples_per_frame)
    batch_size = 2
    slice_size = 75
    slice_pad = 0
    device = torch.device('cpu')
    loader = DataLoader(
        data, batch_size, device, use_predicted=False, slice_size=slice_size, slice_pad=slice_pad)
    assert len(loader) == 1
    item = next(iter(loader))
    assert item.signal_mask[0].sum() <= slice_size * batch_size * samples_per_frame

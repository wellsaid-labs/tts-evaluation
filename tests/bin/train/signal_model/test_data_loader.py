from unittest import mock

import torch
import numpy as np

from src.bin.train.signal_model.data_loader import _get_slice
from src.bin.train.signal_model.data_loader import DataLoader
from src.audio import combine_signal

from tests.utils import get_example_spectrogram_text_speech_rows


@mock.patch('src.bin.train.signal_model.data_loader.random.randint')
def test__get_slice(randint_mock):
    randint_mock.return_value = 5
    samples_per_frame = 10
    spectrogram_channels = 80
    spectrogram = torch.rand(10, spectrogram_channels)
    signal = torch.rand(100)
    slice_pad = 3
    slice_size = 3
    slice_ = _get_slice(
        spectrogram, signal, spectrogram_slice_size=slice_size, spectrogram_slice_pad=slice_pad)

    assert slice_.input_spectrogram.shape == (slice_size + slice_pad * 2, spectrogram_channels)
    assert slice_.input_signal.shape == (slice_size * samples_per_frame, 2)
    assert slice_.target_signal_coarse.shape == (slice_size * samples_per_frame,)
    assert slice_.target_signal_fine.shape == (slice_size * samples_per_frame,)


@mock.patch('src.bin.train.signal_model.data_loader.random.randint')
def test__get_slice__padding(randint_mock):
    randint_mock.return_value = 2
    spectrogram = torch.tensor([[1], [2], [3]])
    signal = torch.tensor([.1, .1, .2, .2, .3, .3])

    slice_pad = 3
    slice_size = 2
    slice_ = _get_slice(
        spectrogram, signal, spectrogram_slice_size=slice_size, spectrogram_slice_pad=slice_pad)

    assert torch.equal(slice_.input_spectrogram,
                       torch.tensor([[0], [1], [2], [3], [0], [0], [0], [0]]))
    target_signal = combine_signal(slice_.target_signal_coarse, slice_.target_signal_fine)
    np.testing.assert_array_almost_equal(
        target_signal.numpy(), torch.tensor([0.3, 0.3, 0, 0]).numpy(), decimal=4)
    source_signal = combine_signal(slice_.input_signal[:, 0], slice_.input_signal[:, 1])
    np.testing.assert_array_almost_equal(
        source_signal.numpy(), torch.tensor([.2, .3, .3, 0.0]).numpy(), decimal=4)
    np.testing.assert_array_almost_equal(slice_.signal_mask, torch.tensor([1, 1, 0, 0]), decimal=4)


def test_data_loader():
    samples_per_frame = 2
    data = get_example_spectrogram_text_speech_rows(samples_per_frame=samples_per_frame)
    batch_size = 2
    slice_size = 75
    slice_pad = 0
    device = torch.device('cpu')
    loader = DataLoader(
        data,
        batch_size,
        device,
        use_predicted=False,
        spectrogram_slice_size=slice_size,
        spectrogram_slice_pad=slice_pad,
        use_tqdm=False,
        trial_run=False,
        num_epochs=2)
    assert len(loader) == 2
    item = next(iter(loader))
    assert item.signal_mask.sum() <= slice_size * batch_size * samples_per_frame

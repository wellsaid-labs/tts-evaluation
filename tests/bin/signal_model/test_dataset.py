from pathlib import Path

import mock
import torch

from src.bin.signal_model._dataset import SignalDataset


@mock.patch('src.bin.signal_model._dataset.random.randint')
def test_signal_dataset_preprocess(randint_mock):
    randint_mock.return_value = 5
    samples_per_frame = 10
    spectrogram_channels = 80
    log_mel_spectrogram = torch.rand(10, spectrogram_channels)
    signal = torch.rand(100)
    frame_pad = 3
    frame_size = 3
    dataset = SignalDataset(source=Path('.'), frame_size=frame_size, frame_pad=frame_pad)
    slice_ = dataset._get_slice(log_mel_spectrogram, signal)

    assert slice_['log_mel_spectrogram'].shape == (frame_size + frame_pad * 2, spectrogram_channels)
    assert slice_['input_signal'].shape == (frame_size * samples_per_frame, 2)
    assert slice_['target_signal_coarse'].shape == (frame_size * samples_per_frame,)
    assert slice_['target_signal_fine'].shape == (frame_size * samples_per_frame,)

from collections import Counter
from unittest import mock

import pytest
import torch

from run.train.signal_model._data import _get_slice


@mock.patch("run.train.signal_model._data.random.randint")
def test__get_slice(randint_mock):
    randint_mock.return_value = 5
    samples_per_frame = 10
    spectrogram_channels = 80
    spectrogram = torch.rand(10, spectrogram_channels)
    signal = torch.rand(100)
    slice_pad = 3
    slice_size = 3
    slice_ = _get_slice(
        spectrogram, signal, spectrogram_slice_size=slice_size, spectrogram_slice_pad=slice_pad
    )

    assert slice_.spectrogram.shape == (slice_size + slice_pad * 2, spectrogram_channels)
    assert slice_.spectrogram_mask.shape == (slice_size + slice_pad * 2,)
    assert slice_.target_signal.shape == (slice_size * samples_per_frame,)
    assert slice_.signal_mask.shape == (slice_size * samples_per_frame,)


def test__get_slice__distribution():
    """ Test that `_get_slice` samples each sample equally. """
    spectrogram = torch.arange(1, 5).unsqueeze(1)
    signal = torch.arange(1, 13)
    slice_size = 3
    spectrogram_slice_pad = 3
    samples = 10000
    sample_counter = Counter()
    frame_counter = Counter()

    for i in range(samples):
        slice_ = _get_slice(
            spectrogram,
            signal,
            spectrogram_slice_size=slice_size,
            spectrogram_slice_pad=spectrogram_slice_pad,
        )
        sample_counter.update(slice_.target_signal.tolist())
        frame_counter.update(slice_.spectrogram.squeeze().tolist())

    total_samples = sum(sample_counter.values()) - sample_counter[0]  # Remove padding
    for i in range(signal.shape[0]):
        # Each sample should be sampled `1 / signal.shape[0]` times
        assert sample_counter[signal[i].item()] / total_samples == pytest.approx(
            1 / signal.shape[0], rel=1e-1
        )

    total_frames = sum(frame_counter.values()) - frame_counter[0]  # Remove padding
    for i in range(spectrogram.shape[0]):
        assert frame_counter[spectrogram[i, 0].item()] / total_frames == pytest.approx(
            1 / spectrogram.shape[0], rel=1e-1
        )


@mock.patch("run.train.signal_model._data.random.randint")
def test__get_slice__padding(randint_mock):
    randint_mock.return_value = 1
    spectrogram = torch.tensor([[1], [2], [3]])
    signal = torch.tensor([0.1, 0.1, 0.2, 0.2, 0.3, 0.3])

    slice_pad = 3
    slice_size = 2
    slice_ = _get_slice(
        spectrogram, signal, spectrogram_slice_size=slice_size, spectrogram_slice_pad=slice_pad
    )

    assert torch.equal(slice_.spectrogram, torch.tensor([[0], [0], [1], [2], [3], [0], [0], [0]]))
    assert torch.equal(
        slice_.spectrogram_mask, torch.tensor([0, 0, 1, 1, 1, 0, 0, 0], dtype=torch.bool)
    )
    assert torch.equal(slice_.target_signal, torch.tensor([0.2, 0.2, 0.3, 0.3]))
    assert torch.equal(slice_.signal_mask, torch.tensor([1, 1, 1, 1], dtype=torch.bool))

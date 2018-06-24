import mock
import numpy as np
import torch

from src.audio import mu_law
from src.audio import mu_law_decode
from src.bin.signal_model._dataset import SignalDataset


@mock.patch('src.bin.signal_model._dataset.random.randint')
def test_signal_dataset_preprocess(randint_mock):
    randint_mock.return_value = 5
    samples_per_frame = 10
    spectrogram_channels = 80
    log_mel_spectrogram = torch.rand(10, spectrogram_channels)
    signal = torch.rand(100)
    receptive_field_size = samples_per_frame
    slice_size = 30
    dataset = SignalDataset(
        source='.', slice_size=slice_size, receptive_field_size=receptive_field_size)
    preprocessed = dataset._preprocess(log_mel_spectrogram, signal)
    assert preprocessed['log_mel_spectrogram'].shape == log_mel_spectrogram.shape
    assert preprocessed['signal'].shape == signal.shape
    assert preprocessed['source_signal_slice'].shape == (slice_size + receptive_field_size,)
    assert preprocessed['target_signal_slice'].shape == (slice_size,)
    np.testing.assert_allclose(preprocessed['source_signal_slice'][receptive_field_size + 1:],
                               mu_law(mu_law_decode(preprocessed['target_signal_slice'][:-1])))
    assert preprocessed['frames_slice'].shape == ((
        slice_size + receptive_field_size) / samples_per_frame, spectrogram_channels)


@mock.patch('src.bin.signal_model._dataset.random.randint')
def test_signal_dataset_preprocess_no_context(randint_mock):
    randint_mock.return_value = 5
    samples_per_frame = 10
    spectrogram_channels = 80
    log_mel_spectrogram = torch.rand(10, spectrogram_channels)
    signal = torch.rand(100)
    receptive_field_size = samples_per_frame
    slice_size = 30
    dataset = SignalDataset(
        source='.',
        slice_size=slice_size,
        receptive_field_size=receptive_field_size,
        add_context=False)
    preprocessed = dataset._preprocess(log_mel_spectrogram, signal)
    assert preprocessed['log_mel_spectrogram'].shape == log_mel_spectrogram.shape
    assert preprocessed['signal'].shape == signal.shape
    assert preprocessed['source_signal_slice'].shape == (slice_size,)
    assert preprocessed['target_signal_slice'].shape == (slice_size,)
    np.testing.assert_allclose(preprocessed['source_signal_slice'][1:],
                               mu_law(mu_law_decode(preprocessed['target_signal_slice'][:-1])))
    assert preprocessed['frames_slice'].shape == (slice_size / samples_per_frame,
                                                  spectrogram_channels)


@mock.patch('src.bin.signal_model._dataset.random.randint')
def test_signal_dataset_preprocess_pad(randint_mock):
    randint_mock.return_value = 1
    samples_per_frame = 10
    spectrogram_channels = 80
    log_mel_spectrogram = torch.rand(10, spectrogram_channels)
    signal = torch.rand(100)
    receptive_field_size = samples_per_frame * 2  # Requires 10 samples of padding
    slice_size = 30
    dataset = SignalDataset(
        source='.', slice_size=slice_size, receptive_field_size=receptive_field_size)
    preprocessed = dataset._preprocess(log_mel_spectrogram, signal)
    assert preprocessed['log_mel_spectrogram'].shape == log_mel_spectrogram.shape
    assert preprocessed['signal'].shape == signal.shape
    assert preprocessed['source_signal_slice'].shape == (slice_size + receptive_field_size,)
    assert preprocessed['target_signal_slice'].shape == (slice_size,)
    np.testing.assert_allclose(preprocessed['source_signal_slice'][receptive_field_size + 1:],
                               mu_law(mu_law_decode(preprocessed['target_signal_slice'][:-1])))
    assert preprocessed['frames_slice'].shape == ((
        slice_size + receptive_field_size) / samples_per_frame, spectrogram_channels)


@mock.patch('src.bin.signal_model._dataset.random.randint')
def test_signal_dataset_preprocess_pad_no_context(randint_mock):
    randint_mock.return_value = 1
    samples_per_frame = 10
    spectrogram_channels = 80
    log_mel_spectrogram = torch.rand(10, spectrogram_channels)
    signal = torch.rand(100)
    receptive_field_size = samples_per_frame * 2  # Requires 10 samples of padding
    slice_size = 30
    dataset = SignalDataset(
        source='.',
        slice_size=slice_size,
        receptive_field_size=receptive_field_size,
        add_context=False)
    preprocessed = dataset._preprocess(log_mel_spectrogram, signal)
    assert preprocessed['log_mel_spectrogram'].shape == log_mel_spectrogram.shape
    assert preprocessed['signal'].shape == signal.shape
    assert preprocessed['source_signal_slice'].shape == (slice_size,)
    assert preprocessed['target_signal_slice'].shape == (slice_size,)
    np.testing.assert_allclose(preprocessed['source_signal_slice'][1:],
                               mu_law(mu_law_decode(preprocessed['target_signal_slice'][:-1])))
    assert preprocessed['frames_slice'].shape == (slice_size / samples_per_frame,
                                                  spectrogram_channels)


@mock.patch('src.bin.signal_model._dataset.random.randint')
def test_signal_dataset_preprocess_receptive_field_size_rounding(randint_mock):
    randint_mock.return_value = 1
    samples_per_frame = 10
    spectrogram_channels = 80
    log_mel_spectrogram = torch.rand(10, spectrogram_channels)
    signal = torch.rand(100)
    receptive_field_size = samples_per_frame * 2 + 2  # Requires 10 samples of padding
    receptive_field_size_rounded = 30
    slice_size = 30
    dataset = SignalDataset(
        source='.', slice_size=slice_size, receptive_field_size=receptive_field_size)
    preprocessed = dataset._preprocess(log_mel_spectrogram, signal)
    assert preprocessed['log_mel_spectrogram'].shape == log_mel_spectrogram.shape
    assert preprocessed['signal'].shape == signal.shape
    assert preprocessed['source_signal_slice'].shape == (slice_size + receptive_field_size_rounded,)
    assert preprocessed['target_signal_slice'].shape == (slice_size,)
    np.testing.assert_allclose(
        preprocessed['source_signal_slice'][-preprocessed['target_signal_slice'].shape[0] + 1:],
        mu_law(mu_law_decode(preprocessed['target_signal_slice'][:-1])))
    assert preprocessed['frames_slice'].shape == ((
        slice_size + receptive_field_size_rounded) / samples_per_frame, spectrogram_channels)
    assert preprocessed['frames_slice'][0:1].sum() == 0
    assert preprocessed['frames_slice'][2].sum() != 0

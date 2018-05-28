import mock
import numpy as np
import os
import torch

from src.optimizer import Optimizer

from src.bin.signal_model._utils import DataIterator
from src.bin.signal_model._utils import load_checkpoint
from src.bin.signal_model._utils import save_checkpoint
from src.bin.signal_model._utils import set_hparams
from src.bin.signal_model._utils import SignalDataset
from src.bin.signal_model._utils import load_data
from src.signal_model import SignalModel
from src.utils.experiment_context_manager import ExperimentContextManager


@mock.patch('src.bin.signal_model._utils.random.randint')
def test_signal_dataset_preprocess(randint_mock):
    randint_mock.return_value = 5
    samples_per_frame = 10
    spectrogram_channels = 80
    log_mel_spectrogram = torch.rand(10, spectrogram_channels)
    quantized_signal = torch.rand(100)
    receptive_field_size = samples_per_frame
    slice_size = 30
    dataset = SignalDataset(
        source='.', slice_size=slice_size, receptive_field_size=receptive_field_size)
    preprocessed = dataset._preprocess(log_mel_spectrogram, quantized_signal)
    assert preprocessed['log_mel_spectrogram'].shape == log_mel_spectrogram.shape
    assert preprocessed['quantized_signal'].shape == quantized_signal.shape
    assert preprocessed['source_signal_slice'].shape == (slice_size + receptive_field_size,)
    assert preprocessed['target_signal_slice'].shape == (slice_size,)
    np.testing.assert_allclose(preprocessed['source_signal_slice'][receptive_field_size + 1:],
                               preprocessed['target_signal_slice'][:-1])
    assert preprocessed['frames_slice'].shape == ((
        slice_size + receptive_field_size) / samples_per_frame, spectrogram_channels)


@mock.patch('src.bin.signal_model._utils.random.randint')
def test_signal_dataset_preprocess_pad(randint_mock):
    randint_mock.return_value = 1
    samples_per_frame = 10
    spectrogram_channels = 80
    log_mel_spectrogram = torch.rand(10, spectrogram_channels)
    quantized_signal = torch.rand(100)
    receptive_field_size = samples_per_frame * 2  # Requires 10 samples of padding
    slice_size = 30
    dataset = SignalDataset(
        source='.', slice_size=slice_size, receptive_field_size=receptive_field_size)
    preprocessed = dataset._preprocess(log_mel_spectrogram, quantized_signal)
    assert preprocessed['log_mel_spectrogram'].shape == log_mel_spectrogram.shape
    assert preprocessed['quantized_signal'].shape == quantized_signal.shape
    assert preprocessed['source_signal_slice'].shape == (slice_size + receptive_field_size,)
    assert preprocessed['target_signal_slice'].shape == (slice_size,)
    np.testing.assert_allclose(preprocessed['source_signal_slice'][receptive_field_size + 1:],
                               preprocessed['target_signal_slice'][:-1])
    assert preprocessed['frames_slice'].shape == ((
        slice_size + receptive_field_size) / samples_per_frame, spectrogram_channels)


@mock.patch('src.bin.signal_model._utils.random.randint')
def test_signal_dataset_preprocess_receptive_field_size_rounding(randint_mock):
    randint_mock.return_value = 1
    samples_per_frame = 10
    spectrogram_channels = 80
    log_mel_spectrogram = torch.rand(10, spectrogram_channels)
    quantized_signal = torch.rand(100)
    receptive_field_size = samples_per_frame * 2 + 2  # Requires 10 samples of padding
    receptive_field_size_rounded = 30
    slice_size = 30
    dataset = SignalDataset(
        source='.', slice_size=slice_size, receptive_field_size=receptive_field_size)
    preprocessed = dataset._preprocess(log_mel_spectrogram, quantized_signal)
    assert preprocessed['log_mel_spectrogram'].shape == log_mel_spectrogram.shape
    assert preprocessed['quantized_signal'].shape == quantized_signal.shape
    assert preprocessed['source_signal_slice'].shape == (slice_size + receptive_field_size_rounded,)
    assert preprocessed['target_signal_slice'].shape == (slice_size,)
    np.testing.assert_allclose(
        preprocessed['source_signal_slice'][-preprocessed['target_signal_slice'].shape[0] + 1:],
        preprocessed['target_signal_slice'][:-1])
    assert preprocessed['frames_slice'].shape == ((
        slice_size + receptive_field_size_rounded) / samples_per_frame, spectrogram_channels)


def test_load_data():
    train, dev = load_data(
        source_train='tests/_test_data/signal_dataset/train',
        source_dev='tests/_test_data/signal_dataset/dev',
        log_mel_spectrogram_prefix='log_mel_spectrogram',
        quantized_signal_prefix='quantized_signal',
        extension='.npy')
    assert len(train) == 1
    assert len(dev) == 1

    # Smoke test
    dev[0]


def test_set_hparams():
    set_hparams()
    model = SignalModel()
    optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()))
    assert optimizer.defaults['eps'] == 10**-8


def test_data_iterator():
    with ExperimentContextManager(label='test_data_iterator') as context:
        dataset = [{
            'source_signal_slice': torch.randint(low=0, high=255, size=(100,)),
            'target_signal_slice': torch.randint(low=0, high=255, size=(100,)),
            'frames_slice': torch.FloatTensor(10, 80),
            'log_mel_spectrogram': torch.FloatTensor(30, 80),
        }, {
            'source_signal_slice': torch.randint(low=0, high=255, size=(100,)),
            'target_signal_slice': torch.randint(low=0, high=255, size=(100,)),
            'frames_slice': torch.FloatTensor(10, 80),
            'log_mel_spectrogram': torch.FloatTensor(30, 80),
        }]
        batch_size = 1

        iterator = DataIterator(context.device, dataset, batch_size)
        assert len(iterator) == 2
        next(iter(iterator))

        iterator = DataIterator(context.device, dataset, batch_size, trial_run=True)
        assert len(iterator) == 1
        iterator = iter(iterator)
        next(iterator)
        try:
            next(iterator)
        except StopIteration:
            error = True
        assert error


def test_load_save_checkpoint():
    with ExperimentContextManager(label='test_load_save_checkpoint') as context:
        model = SignalModel()
        optimizer = Optimizer(
            torch.optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters())))
        filename = save_checkpoint(
            context.checkpoints_directory, model=model, optimizer=optimizer, step=10)
        assert os.path.isfile(filename)

        # Smoke test
        load_checkpoint(filename)

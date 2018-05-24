import mock
import os
import torch

from torchnlp.datasets import Dataset
from torch.optim.lr_scheduler import StepLR
from src.optimizer import Optimizer

from src.bin.feature_model._utils import DataIterator
from src.bin.feature_model._utils import load_checkpoint
from src.bin.feature_model._utils import load_data
from src.bin.feature_model._utils import save_checkpoint
from src.feature_model import FeatureModel
from src.utils.experiment_context_manager import ExperimentContextManager


def test_data_iterator():
    with ExperimentContextManager(label='test_data_iterator') as context:
        dataset = [{
            'text': torch.LongTensor([1, 2, 3]),
            'log_mel_spectrogram': torch.FloatTensor(500, 80),
            'stop_token': torch.FloatTensor(500)
        }, {
            'text': torch.LongTensor([1, 2]),
            'log_mel_spectrogram': torch.FloatTensor(450, 80),
            'stop_token': torch.FloatTensor(450)
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


@mock.patch('src.bin.feature_model._utils.lj_speech_dataset')
def test_load_data(lj_speech_dataset_mock):
    cache = 'tests/_test_data/lj_speech.pt'
    signal_cache = 'tests/_test_data/lj_speech_signals.pt'

    with ExperimentContextManager(label='test_load_data') as context:
        lj_speech_dataset_mock.return_value = Dataset([{
            'text': 'Printing, in the only sense with which we are at present concerned,...',
            'wav': 'tests/_test_data/LJ001-0001.wav'
        }, {
            'text': 'in being comparatively modern.',
            'wav': 'tests/_test_data/LJ001-0002.wav'
        }])
        train, dev, encoder = load_data(
            context,
            cache=cache,
            signal_cache=signal_cache,
            splits=(0.5, 0.5),
            load_signal=True,
            use_multiprocessing=False)
        assert os.path.isfile(cache)
        assert len(train) == 1
        assert len(dev) == 1

        assert train[0]['stop_token'].shape[0] == train[0]['log_mel_spectrogram'].shape[0]
        assert train[0]['quantized_signal'].shape[0] % train[0]['log_mel_spectrogram'].shape[0] == 0

        # Test Cache
        train, dev, encoder = load_data(
            context,
            cache=cache,
            signal_cache=signal_cache,
            load_signal=True,
            use_multiprocessing=False)
        lj_speech_dataset_mock.assert_called_once()
        assert len(train) == 1
        assert len(dev) == 1

    # Clean up
    os.remove(cache)
    os.remove(signal_cache)


def test_load_save_checkpoint():
    with ExperimentContextManager(label='test_load_save_checkpoint') as context:
        model = FeatureModel(10)
        optimizer = Optimizer(
            torch.optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters())))
        scheduler = StepLR(optimizer.optimizer, step_size=30)
        filename = save_checkpoint(
            context.checkpoints_directory,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            step=10)
        assert os.path.isfile(filename)

        # Smoke test
        load_checkpoint(filename)

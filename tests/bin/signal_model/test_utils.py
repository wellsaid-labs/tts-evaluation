import os
import torch

from src.optimizer import Optimizer

from src.bin.signal_model._utils import DataIterator
from src.bin.signal_model._utils import load_checkpoint
from src.bin.signal_model._utils import save_checkpoint
from src.bin.signal_model._utils import set_hparams
from src.signal_model import SignalModel
from src.utils.experiment_context_manager import ExperimentContextManager


def test_set_hparams():
    set_hparams()
    model = SignalModel()
    optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()))
    assert optimizer.defaults['eps'] == 10**-8


def test_data_iterator():
    with ExperimentContextManager(label='test_data_iterator') as context:
        dataset = [{
            'quantized_signal': torch.randint(low=0, high=255, size=(900,)),
            'log_mel_spectrogram': torch.FloatTensor(30, 80),
        }, {
            'quantized_signal': torch.randint(low=0, high=255, size=(300,)),
            'log_mel_spectrogram': torch.FloatTensor(10, 80),
        }]
        batch_size = 1

        iterator = DataIterator(context.device, dataset, batch_size, max_samples=600)
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

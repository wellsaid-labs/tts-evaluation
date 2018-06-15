import os
import torch

from src.optimizer import Optimizer

from src.bin.signal_model._utils import load_checkpoint
from src.bin.signal_model._utils import load_data
from src.bin.signal_model._utils import save_checkpoint
from src.bin.signal_model._utils import set_hparams
from src.signal_model import WaveNet
from src.utils.experiment_context_manager import ExperimentContextManager


def test_load_data():
    train, dev = load_data(
        source_train='tests/_test_data/signal_dataset/train',
        source_dev='tests/_test_data/signal_dataset/dev',
        log_mel_spectrogram_prefix='log_mel_spectrogram',
        signal_prefix='signal',
        extension='.npy')
    assert len(train) == 1
    assert len(dev) == 1

    # Smoke test
    dev[0]


def test_set_hparams():
    set_hparams()
    model = WaveNet()
    optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()))
    assert optimizer.defaults['eps'] == 10**-8


def test_load_save_checkpoint():
    with ExperimentContextManager(label='test_load_save_checkpoint') as context:
        model = WaveNet()
        optimizer = Optimizer(
            torch.optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters())))
        filename = save_checkpoint(
            context.checkpoints_directory, model=model, optimizer=optimizer, step=10)
        assert os.path.isfile(filename)

        # Smoke test
        load_checkpoint(filename)

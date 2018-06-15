import mock
import os
import torch

from torchnlp.datasets import Dataset
from torch.optim.lr_scheduler import StepLR
from src.optimizer import Optimizer

from src.bin.feature_model._utils import load_checkpoint
from src.bin.feature_model._utils import load_data
from src.bin.feature_model._utils import save_checkpoint
from src.bin.feature_model._utils import set_hparams
from src.feature_model import FeatureModel
from src.utils.experiment_context_manager import ExperimentContextManager


def test_set_hparams():
    # Smoke test
    set_hparams()


@mock.patch('src.bin.feature_model._utils.lj_speech_dataset')
def test_load_data(lj_speech_dataset_mock):
    lj_speech_dataset_mock.return_value = tuple([
        Dataset([{
            'text': 'Printing, in the only sense with which we are at present concerned,...',
            'wav': 'tests/_test_data/LJ001-0001.wav'
        }]),
        Dataset([{
            'text': 'Printing, in the only sense with which we are at present concerned,...',
            'wav': 'tests/_test_data/LJ001-0001.wav'
        }])
    ])
    train, dev, encoder = load_data(sample_rate=22050)
    assert len(train) == 1
    assert len(dev) == 1
    assert train[0]['stop_token'].shape[0] == train[0]['log_mel_spectrogram'].shape[0]
    assert train[0]['signal'].shape[0] % train[0]['log_mel_spectrogram'].shape[0] == 0


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

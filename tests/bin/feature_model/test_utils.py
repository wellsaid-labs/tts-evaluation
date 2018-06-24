import os
import torch

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


def test_load_data():
    train, dev, text_encoder = load_data(
        source_train='tests/_test_data/feature_dataset/train',
        source_dev='tests/_test_data/feature_dataset/dev')
    assert len(train) == 1
    assert len(dev) == 1
    assert text_encoder.decode(train[0]['text']) == 'Yup!'

    # Smoke test
    dev[0]


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

        context.clean_up()

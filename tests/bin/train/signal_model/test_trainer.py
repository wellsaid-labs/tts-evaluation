from unittest import mock

import pytest
import torch

from src.audio import read_audio
from src.bin.train.signal_model.data_loader import SignalModelTrainingRow
from src.bin.train.signal_model.trainer import ExponentialMovingParameterAverage
from src.bin.train.signal_model.trainer import Trainer
from src.environment import TEMP_PATH
from src.utils import Checkpoint
from tests._utils import get_tts_mocks
from tests._utils import MockCometML


@mock.patch('src.bin.train.signal_model.trainer.atexit.register')
@mock.patch('src.datasets.utils.read_audio')
def get_trainer(read_audio_mock, register_mock, load_data=True):
    """
    Args:
        load_data (bool, optional): If `False` do not load any data for faster instantiation.
    """
    mocks = get_tts_mocks(add_predicted_spectrogram=load_data, add_spectrogram=load_data)

    register_mock.return_value = None
    # NOTE: Decrease the audio size for test performance.
    read_audio_mock.side_effect = lambda *args, **kwargs: read_audio(*args, **kwargs)[:4096]

    return Trainer(
        comet_ml=MockCometML(),
        device=mocks['device'],
        checkpoints_directory=TEMP_PATH,
        train_dataset=mocks['train_dataset'] if load_data else [],
        dev_dataset=mocks['dev_dataset'] if load_data else [],
        spectrogram_model_checkpoint_path=mocks['spectrogram_model_checkpoint'].path,
        train_batch_size=1,
        dev_batch_size=1)


def test_exponential_moving_parameter_average__identity():

    class Module(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.parameter = torch.nn.Parameter(torch.zeros(1))

    module = Module()
    exponential_moving_parameter_average = ExponentialMovingParameterAverage(module, beta=0)

    module.parameter.data[0] = 1.0
    exponential_moving_parameter_average.update()
    exponential_moving_parameter_average.apply_shadow()
    assert module.parameter.data[0] == 1.0
    exponential_moving_parameter_average.restore()

    module.parameter.data[0] = 2.0
    exponential_moving_parameter_average.update()
    exponential_moving_parameter_average.apply_shadow()
    assert module.parameter.data[0] == 2.0
    exponential_moving_parameter_average.restore()


def test_exponential_moving_parameter_average():
    """ Test bias correction implementation via this video:
    https://pt.coursera.org/lecture/deep-neural-network/www.deeplearning.ai-XjuhD
    """
    values = [1.0, 2.0]

    class Module(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.parameter = torch.nn.Parameter(torch.full((2,), values[0]))

    module = Module()
    exponential_moving_parameter_average = ExponentialMovingParameterAverage(module, beta=0.98)

    module.parameter.data = torch.tensor([values[1], values[1]])

    exponential_moving_parameter_average.update()

    assert module.parameter.data[0] == values[1]
    assert module.parameter.data[1] == values[1]

    exponential_moving_parameter_average.apply_shadow()

    assert module.parameter.data[0] == (0.0196 * values[0] + 0.02 * values[1]) / 0.0396
    assert module.parameter.data[1] == (0.0196 * values[0] + 0.02 * values[1]) / 0.0396

    exponential_moving_parameter_average.restore()

    assert module.parameter.data[0] == values[1]
    assert module.parameter.data[1] == values[1]


@mock.patch('src.bin.train.signal_model.trainer.atexit.register')
def test_checkpoint(register_mock):
    """ Ensure checkpoint can be saved and loaded from. """
    trainer = get_trainer(load_data=False)

    register_mock.return_value = None

    checkpoint_path = trainer.save_checkpoint()
    Trainer.from_checkpoint(
        comet_ml=MockCometML(),
        checkpoint=Checkpoint.from_path(checkpoint_path),
        device=torch.device('cpu'),
        checkpoints_directory=trainer.checkpoints_directory,
        train_dataset=trainer.train_dataset,
        dev_dataset=trainer.dev_dataset)


def test__do_loss_and_maybe_backwards():
    """ Test that the correct loss values are computed and back propagated. """
    trainer = get_trainer(load_data=False)
    batch = SignalModelTrainingRow(
        spectrogram_mask=None,
        spectrogram=None,
        target_signal=torch.zeros(2, 4096),
        source_signal=None,
        signal_mask=torch.ones(2, 4096))
    predicted_signal = torch.zeros(2, 4096)

    log_mel_spectrogram_magnitude_loss, num_predictions = trainer._do_loss_and_maybe_backwards(
        batch, predicted_signal, False)
    assert log_mel_spectrogram_magnitude_loss.item() == pytest.approx(0.0)
    assert num_predictions == 8192


def test_visualize_inferred():
    """ Smoke test to ensure that `visualize_inferred` runs without failure. """
    trainer = get_trainer()
    trainer.visualize_inferred()


def test_run_epoch():
    """ Smoke test to ensure that `run_epoch` runs without failure. """
    trainer = get_trainer()
    trainer.run_epoch(train=False, trial_run=False)
    trainer.run_epoch(train=False, trial_run=True)

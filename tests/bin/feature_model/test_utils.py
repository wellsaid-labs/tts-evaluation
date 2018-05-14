import mock
import os
import torch
import shutil

from torchnlp.datasets import Dataset
from torch.optim.lr_scheduler import StepLR
from src.optimizer import Optimizer

from src.bin.feature_model._utils import DataIterator
from src.bin.feature_model._utils import load_checkpoint
from src.bin.feature_model._utils import load_data
from src.bin.feature_model._utils import sample_attention
from src.bin.feature_model._utils import sample_spectrogram
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

        iterator = DataIterator(context, dataset, batch_size)
        assert len(iterator) == 2
        next(iter(iterator))

        iterator = DataIterator(context, dataset, batch_size, trial_run=True)
        assert len(iterator) == 1
        next(iter(iterator))


@mock.patch('src.bin.feature_model._utils.lj_speech_dataset')
def test_load_data(lj_speech_dataset_mock):
    cache = 'tests/_test_data/lj_speech.pt'

    with ExperimentContextManager(label='test_load_data') as context:
        lj_speech_dataset_mock.return_value = Dataset([{
            'text': 'Printing, in the only sense with which we are at present concerned,...',
            'wav': 'data/LJSpeech-1.1/wavs/LJ001-0001.wav'
        }, {
            'text': 'in being comparatively modern.',
            'wav': 'data/LJSpeech-1.1/wavs/LJ001-0002.wav'
        }])
        train, dev, encoder = load_data(context, cache, splits=(0.5, 0.5))
        assert os.path.isfile(cache)
        assert len(train) == 1
        assert len(dev) == 1

        assert train[0]['stop_token'].shape[0] == train[0]['log_mel_spectrogram'].shape[0]
        assert train[0]['quantized_signal'].shape[0] % train[0]['log_mel_spectrogram'].shape[0] == 0

        # Test Cache
        train, dev, encoder = load_data(context, cache)
        lj_speech_dataset_mock.assert_called_once()
        assert len(train) == 1
        assert len(dev) == 1

    # Clean up
    os.remove(cache)
    shutil.rmtree(context.directory)


def test_load_save_checkpoint():
    with ExperimentContextManager(label='test_load_data') as context:
        model = FeatureModel(10)
        optimizer = Optimizer(
            torch.optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters())))
        scheduler = StepLR(optimizer.optimizer, step_size=30)
        filename = save_checkpoint(
            context, model=FeatureModel(10), optimizer=optimizer, scheduler=scheduler)
        assert os.path.isfile(filename)

        # Smoke test
        load_checkpoint(filename)

    # Clean up
    os.remove(filename)
    shutil.rmtree(context.directory)


@mock.patch('src.bin.feature_model._utils.plot_spectrogram', return_value=None)
@mock.patch('src.bin.feature_model._utils.log_mel_spectrogram_to_wav', return_value=None)
def test_sample_spectrogram(log_mel_spectrogram_to_wav_mock, plot_spectrogram_mock):
    # Smoke test
    sample_spectrogram(torch.FloatTensor(4, 4, 80), '')

    plot_spectrogram_mock.assert_called_once()
    log_mel_spectrogram_to_wav_mock.assert_called_once()


@mock.patch('src.bin.feature_model._utils.plot_attention', return_value=None)
def test_sample_attention(plot_attention_mock):
    # Smoke test
    sample_attention(torch.FloatTensor(4, 4, 80), '')
    plot_attention_mock.assert_called_once()

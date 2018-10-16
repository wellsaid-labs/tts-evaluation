import pytest
import torch

from src.bin.signal_model._data_iterator import DataIterator
from src.utils.experiment_context_manager import ExperimentContextManager


@pytest.fixture
def dataset():
    return [{
        'slice': {
            'input_signal': torch.FloatTensor(100, 2),
            'target_signal_coarse': torch.FloatTensor(100,),
            'target_signal_fine': torch.FloatTensor(100,),
            'log_mel_spectrogram': torch.FloatTensor(50, 80),
        },
        'log_mel_spectrogram': torch.FloatTensor(50, 80),
        'signal': torch.FloatTensor(100),
    }, {
        'slice': {
            'input_signal': torch.FloatTensor(90, 2),
            'target_signal_coarse': torch.FloatTensor(90,),
            'target_signal_fine': torch.FloatTensor(90,),
            'log_mel_spectrogram': torch.FloatTensor(50, 80),
        },
        'log_mel_spectrogram': torch.FloatTensor(50, 80),
        'signal': torch.FloatTensor(90),
    }]


def test_data_iterator(dataset):
    """ Smoke test that DataIterator can produce batches """
    with ExperimentContextManager(label='test_data_iterator') as context:
        batch_size = 2
        iterator = DataIterator(context.device, dataset, batch_size)
        assert len(iterator) == 1
        item = next(iter(iterator))

        assert torch.sum(item['slice']['signal_mask']) == 190

        iterator = DataIterator(context.device, dataset, batch_size, trial_run=True)
        assert len(iterator) == 1
        iterator = iter(iterator)
        next(iterator)
        try:
            next(iterator)
        except StopIteration:
            error = True
        assert error

        context.clean_up()

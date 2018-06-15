import torch

from src.bin.feature_model._data_iterator import DataIterator
from src.utils.experiment_context_manager import ExperimentContextManager


def test_data_iterator():
    with ExperimentContextManager(label='test_data_iterator') as context:
        dataset = [{
            'text': torch.LongTensor([1, 2, 3]),
            'log_mel_spectrogram': torch.FloatTensor(500, 80),
            'stop_token': torch.FloatTensor(500),
            'signal': torch.FloatTensor(1200),
        }, {
            'text': torch.LongTensor([1, 2]),
            'log_mel_spectrogram': torch.FloatTensor(450, 80),
            'stop_token': torch.FloatTensor(450),
            'signal': torch.FloatTensor(1000),
        }]
        batch_size = 1

        iterator = DataIterator(context.device, dataset, batch_size, load_signal=True)
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

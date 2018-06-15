import torch

from src.audio import mu_law_encode
from src.audio import mu_law
from src.audio import mu_law_decode
from src.bin.signal_model._data_iterator import DataIterator
from src.utils.experiment_context_manager import ExperimentContextManager


def test_data_iterator():
    with ExperimentContextManager(label='test_data_iterator') as context:
        source_signal_slice = torch.FloatTensor(100,)
        target_signal_slice = mu_law_encode(source_signal_slice)[1:]
        source_signal_slice = mu_law(mu_law_decode(mu_law_encode(source_signal_slice)))
        source_signal_slice = source_signal_slice[:-1]
        dataset = [{
            'source_signal_slice': source_signal_slice,
            'target_signal_slice': target_signal_slice,
            'frames_slice': torch.FloatTensor(10, 80),
            'log_mel_spectrogram': torch.FloatTensor(30, 80),
            'signal': torch.FloatTensor(300),
        }, {
            'source_signal_slice': source_signal_slice,
            'target_signal_slice': target_signal_slice,
            'frames_slice': torch.FloatTensor(10, 80),
            'log_mel_spectrogram': torch.FloatTensor(30, 80),
            'signal': torch.FloatTensor(300),
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

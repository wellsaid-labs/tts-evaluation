from src.bin.feature_model._data_iterator import DataIterator
from src.utils.experiment_context_manager import ExperimentContextManager
from src.bin.feature_model._utils import load_data


def test_data_iterator():
    with ExperimentContextManager(label='test_data_iterator') as context:
        # TODO: For a unit test, this shoulded by mocked
        train, _, _ = load_data(
            source_train='tests/_test_data/feature_dataset/train',
            source_dev='tests/_test_data/feature_dataset/dev')
        batch_size = 1

        iterator = DataIterator(context.device, train, batch_size, load_signal=True)
        assert len(iterator) == 1
        next(iter(iterator))

        iterator = DataIterator(context.device, train, batch_size, trial_run=True)
        assert len(iterator) == 1
        iterator = iter(iterator)
        next(iterator)
        try:
            next(iterator)
        except StopIteration:
            error = True
        assert error

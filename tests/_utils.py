from collections.abc import MutableMapping
from contextlib import contextmanager
from unittest import mock
from functools import lru_cache

import pathlib
import shutil
import urllib.request

from torch.optim import Adam

import torch
import pytest

from src.datasets import add_predicted_spectrogram_column
from src.datasets import add_spectrogram_column
from src.datasets import lj_speech_dataset
from src.datasets import m_ailabs_en_us_speech_dataset
from src.datasets import normalize_audio_column
from src.datasets.m_ailabs import DOROTHY_AND_WIZARD_OZ
from src.hparams import add_config
from src.optimizers import Optimizer
from src.signal_model import WaveRNN
from src.spectrogram_model import InputEncoder
from src.spectrogram_model import SpectrogramModel
from src.utils import AnomalyDetector
from src.utils import Checkpoint
from src.utils import OnDiskTensor
from src.utils import split_list


def create_disk_garbage_collection_fixture(root_directory, **kwargs):
    """ Create fixture for deleting extra files and directories after a test is run.
    """

    @pytest.fixture(**kwargs)
    def fixture():
        all_paths = lambda: set(root_directory.rglob('*')) if root_directory.exists() else set()

        before = all_paths()
        yield root_directory
        after = all_paths()

        for path in after.difference(before):
            if not path.exists():
                continue

            # NOTE: These `print`s will help debug a test if it fails; otherwise, they are ignored.
            if path.is_dir():
                print('Deleting directory: ', path)
                shutil.rmtree(str(path))
            elif path.is_file():
                print('Deleting file: ', path)
                path.unlink()

        assert before == all_paths()

    return fixture


class MockCometML():

    def __init__(self, *args, **kwargs):
        self.project_name = ''

    @contextmanager
    def train(self, *args, **kwargs):
        yield self

    @contextmanager
    def validate(self, *args, **kwargs):
        yield self

    def get_key(self):
        return ''

    def __getattr__(self, attr):
        return lambda *args, **kwargs: self


class MockOnDiskTensor(OnDiskTensor):

    def __init__(self, path, tensor, exists=True):
        self.tensor = tensor
        self.exists = exists
        self.path = path
        self.allow_pickle = False

    @property
    def shape(self):
        return self.tensor.shape

    def to_tensor(self):
        return self.tensor

    def exists(self):
        return self.exists

    def unlink(self):
        return self

    def from_tensor(self, tensor):
        self.tensor = tensor
        return self


# Check the URL requested is valid
def url_first_side_effect(url, *args, **kwargs):
    # TODO: Fix failure case if internet does not work
    assert urllib.request.urlopen(url).getcode() == 200
    return None


# NOTE: Consumes the first argument
url_second_side_effect = lambda _, *args, **kwargs: url_first_side_effect(*args, **kwargs)


# Learn more about `Mapping`:
# http://www.kr41.net/2016/03-23-dont_inherit_python_builtin_dict_type.html
# https://treyhunner.com/2019/04/why-you-shouldnt-inherit-from-list-and-dict-in-python/
class LazyDict(MutableMapping):
    """ Lazy dictionary such that each value is a callable that's executed and cached on
    `__getitem__`.
    """

    def __init__(self, *args, **kwargs):
        self._dict = dict(*args, **kwargs)
        assert all(callable(v) for v in self._dict.values()), 'All values must be callables.'
        self._cache = dict()

    def __getitem__(self, key):
        if key in self._cache:
            return self._cache[key]

        results = self._dict.__getitem__(key)()
        self._cache[key] = results
        return results

    def __setitem__(self, key, value):
        assert callable(value), 'All values must be callables.'
        self._dict[key] = value

    def __delitem__(self, key):
        del self._dict[key]
        del self._cache[key]

    def __iter__(self):
        return iter(self._dict)

    def __len__(self):
        return len(self._dict)


TEST_DATA_PATH = pathlib.Path('tests/_test_data/_utils')


@lru_cache()
def _get_mock_data():
    """ Get a mock dataset for testing. """
    with mock.patch('urllib.request.urlretrieve') as mock_urlretrieve:
        mock_urlretrieve.side_effect = url_first_side_effect
        data = m_ailabs_en_us_speech_dataset(
            directory=TEST_DATA_PATH / 'M-AILABS', all_books=[DOROTHY_AND_WIZARD_OZ])
        data += lj_speech_dataset(directory=TEST_DATA_PATH)
    return data


def get_tts_mocks(add_spectrogram=False,
                  add_predicted_spectrogram=False,
                  add_spectrogram_kwargs={},
                  add_predicted_spectrogram_kwargs={}):
    """ Get mock data for integration testing the TTS pipeline.

    Args:
        add_spectrogram (bool): Compute the spectrogram for the dataset.
        add_predicted_spectrogram (bool): Compute the predicted spectrogram for the dataset.
        add_spectrogram_kwargs (dict): `kwargs` passed onto `add_spectrogram_column`.
        add_predicted_spectrogram_kwargs (dict): `kwargs` passed onto
            `add_predicted_spectrogram_column`.

    Returns:
        (LazyDict): The dict has various objects required to run an integration test.
    """
    return_ = LazyDict({})

    return_['device'] = lambda: torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def get_dataset():
        dataset = normalize_audio_column(_get_mock_data())

        if add_spectrogram:
            dataset = add_spectrogram_column(dataset, **add_spectrogram_kwargs)

        if add_predicted_spectrogram:
            dataset = add_predicted_spectrogram_column(dataset,
                                                       return_['spectrogram_model_checkpoint'],
                                                       return_['device'],
                                                       **add_predicted_spectrogram_kwargs)

        return dataset

    return_['dataset'] = get_dataset
    return_['input_encoder'] = lambda: InputEncoder([e.text for e in _get_mock_data()],
                                                    [e.speaker for e in _get_mock_data()])

    def get_spectrogram_model():
        # NOTE: Configure the `SpectrogramModel` to stop iteration as soon as possible.
        add_config({'src.spectrogram_model.model.SpectrogramModel._infer.stop_threshold': 0.0})
        return SpectrogramModel(return_['input_encoder'].text_encoder.vocab_size,
                                return_['input_encoder'].speaker_encoder.vocab_size)

    return_['spectrogram_model'] = get_spectrogram_model

    def get_spectrogram_model_checkpoint():
        parameters = return_['spectrogram_model'].parameters()
        spectrogram_model_optimizer = Optimizer(
            Adam(params=filter(lambda p: p.requires_grad, parameters)))
        checkpoint = Checkpoint(
            comet_ml_project_name='',
            comet_ml_experiment_key='',
            directory=TEST_DATA_PATH,
            model=return_['spectrogram_model'],
            optimizer=spectrogram_model_optimizer,
            epoch=0,
            step=0,
            input_encoder=return_['input_encoder'])
        checkpoint.save()
        return checkpoint

    return_['spectrogram_model_checkpoint'] = get_spectrogram_model_checkpoint
    return_['signal_model'] = lambda: WaveRNN()

    def get_signal_model_checkpoint():
        signal_model_optimizer = Optimizer(
            Adam(params=filter(lambda p: p.requires_grad, return_['signal_model'].parameters())))
        checkpoint = Checkpoint(
            comet_ml_project_name='',
            comet_ml_experiment_key='',
            directory=TEST_DATA_PATH,
            epoch=0,
            step=0,
            anomaly_detector=AnomalyDetector(),
            optimizer=signal_model_optimizer,
            model=return_['signal_model'],
            num_rollbacks=0,
            spectrogram_model_checkpoint_path=return_['spectrogram_model_checkpoint'].path)
        checkpoint.save()
        return checkpoint

    return_['signal_model_checkpoint'] = get_signal_model_checkpoint
    return_['train_dataset'] = lambda: split_list(return_['dataset'], (0.5, 0.5))[0]
    return_['dev_dataset'] = lambda: split_list(return_['dataset'], (0.5, 0.5))[1]
    return return_

from collections import namedtuple
from pathlib import Path
from unittest import mock

import logging
import os
import random
import sys

from torch import nn
from torchnlp.utils import collate_tensors

import numpy
import pytest
import torch

from src.datasets import Gender
from src.datasets import Speaker
from src.datasets import TextSpeechRow
from src.optimizers import Optimizer
from src.spectrogram_model import InputEncoder
from src.spectrogram_model import SpectrogramModel
from src.utils import AccumulatedMetrics
from src.utils import align_tokens
from src.utils import AnomalyDetector
from src.utils import assert_enough_disk_space
from src.utils import balance_list
from src.utils import batch_predict_spectrograms
from src.utils import Checkpoint
from src.utils import DataLoader
from src.utils import DataLoaderDataset
from src.utils import dict_collapse
from src.utils import duplicate_stream
from src.utils import evaluate
from src.utils import flatten_parameters
from src.utils import format_alignment
from src.utils import get_average_norm
from src.utils import get_random_generator_state
from src.utils import get_tensors_dim_length
from src.utils import get_total_parameters
from src.utils import get_weighted_stdev
from src.utils import identity
from src.utils import log_runtime
from src.utils import OnDiskTensor
from src.utils import parse_hparam_args
from src.utils import ROOT_PATH
from src.utils import set_basic_logging_config
from src.utils import set_random_generator_state
from src.utils import slice_by_cumulative_sum
from src.utils import sort_together
from src.utils import split_list


def test_slice_by_cumulative_sum():
    assert [1, 2, 3] == slice_by_cumulative_sum([1, 2, 3, 4], max_total_value=6)
    assert [(1, 1), (1, 2)] == slice_by_cumulative_sum([(1, 1), (1, 2), (1, 3), (1, 4)],
                                                       max_total_value=4,
                                                       get_value=lambda x: x[1])


def test_assert_enough_disk_space__smoke_test():
    assert_enough_disk_space(0)


def test_sort_together():
    assert ['c', 'a', 'b'] == sort_together(['a', 'b', 'c'], [2, 3, 1])


def test_get_tensors_dim_length():
    assert [5, 5] == get_tensors_dim_length([torch.randn(5, 5) for _ in range(2)],
                                            dim=1,
                                            use_tqdm=True)


def test_log_runtime__smoke_test():
    func = lambda x: x + 1
    func = log_runtime(func)
    func(1)


def test_set_basic_logging_config__smoke_test():
    set_basic_logging_config()
    set_basic_logging_config(torch.device('cpu'))


def test_dict_collapse():
    assert {
        'a': 1,
        'b.c': 2,
        'b.d.e': 3,
        'g': []
    } == dict_collapse({
        'a': 1,
        'b': {
            'c': 2,
            'd': {
                'e': 3
            }
        },
        'f': {},
        'g': []
    })


def test_on_disk_tensor():
    original = torch.rand(4, 10)
    tensor = OnDiskTensor.from_tensor('tests/_test_data/tensor.npy', original)
    assert tensor.shape == original.shape
    assert tensor._shape is not None
    assert tensor.shape == original.shape  # Smoke test caching
    assert torch.equal(tensor.to_tensor(), original)
    assert tensor.exists()
    tensor.unlink()


def test_on_disk_tensor_eq():
    assert OnDiskTensor('tests/_test_data/tensor.npy') == OnDiskTensor(
        'tests/_test_data/tensor.npy')
    assert OnDiskTensor('tests/_test_data/other_tensor.npy') != OnDiskTensor(
        'tests/_test_data/tensor.npy')


def test_on_disk_tensor_hash():
    assert hash(OnDiskTensor('tests/_test_data/tensor.npy')) == hash(
        OnDiskTensor('tests/_test_data/tensor.npy'))


def test_data_loader_dataset():
    expected = [2, 3]
    dataset = DataLoaderDataset([1, 2], lambda x: x + 1)
    assert len(dataset) == len(expected)
    assert list(dataset) == expected


def test_data_loader():
    dataset = [1]
    for batch in DataLoader(
            dataset,
            trial_run=True,
            post_processing_fn=lambda x: x + 1,
            load_fn=lambda x: x + 1,
            num_workers=1,
            use_tqdm=True):
        assert len(batch) == 1
        assert batch[0] == 3


MockTuple = namedtuple('MockTuple', ['t'])


def test_data_loader__named_tuple__collate_fn():
    dataset = [MockTuple(t=torch.Tensor(1)), MockTuple(t=torch.Tensor(1))]
    for batch in DataLoader(dataset, num_workers=1, batch_size=2, collate_fn=collate_tensors):
        assert batch.t.shape == (2, 1)


def test_evaluate():
    model = nn.LSTM(10, 10).train(mode=True)
    model_two = nn.LSTM(10, 10).train(mode=True)
    with evaluate(model, model_two, device=torch.device('cpu')):
        # NOTE: Unable to test ``no_grad`` without a PyTorch helper function
        # NOTE: Unable to test device change without a secondary device available.
        assert not model.training
        assert not model_two.training
    assert model.training
    assert model_two.training


def test_identity():
    assert identity(2) == 2


@pytest.fixture
def stdout_log():
    path = Path('tests/_test_data/stdout.log')

    yield path

    if path.is_file():
        path.unlink()


def test_duplicate_stream(capsys, stdout_log):
    assert not stdout_log.is_file()  # Ensure everything is clean up

    with capsys.disabled():  # Disable capsys because it messes with sys.stdout
        logger = logging.getLogger(__name__)
        handler = logging.StreamHandler(sys.stdout)
        logger.addHandler(handler)
        stop = duplicate_stream(sys.stdout, stdout_log)

        print('1')
        logger.info('2')
        os.system('echo 3')

        # Flush and close
        stop()
        logger.removeHandler(handler)

    assert stdout_log.is_file()
    output = stdout_log.read_text()
    assert set(output.split()) == set(['1', '2', '3'])


def test_get_average_norm__shape_invariant():
    tensor = torch.Tensor([0.5, 0.2, 0.3])
    expanded = tensor.unsqueeze(1).expand(3, 4)
    assert get_average_norm(tensor) == get_average_norm(expanded)


def test_get_average_norm__mask_invariant():
    tensor = torch.randn(3, 4, 5)
    mask = torch.ones(3, 5).byte()
    assert get_average_norm(tensor, dim=1) == get_average_norm(tensor, mask=mask, dim=1)


def test_get_weighted_stdev__constant():
    tensor = torch.Tensor([0, 1, 0])
    standard_deviation = get_weighted_stdev(tensor, dim=0)
    assert standard_deviation == 0.0  # 0.0 is the standard deviation for a constant


def test_get_weighted_stdev__bias():
    tensor = torch.Tensor([.25, .25, .25, .25])
    standard_deviation = get_weighted_stdev(tensor, dim=0)
    # Population standard deviation for 1, 2, 3, 4
    assert standard_deviation == pytest.approx(1.1180339887499)


def test_get_weighted_stdev():
    tensor = torch.Tensor([[[0.33333, 0.33333, 0.33334], [0, 0.5, 0.5]],
                           [[0, 0.5, 0.5], [0, 0.5, 0.5]]])
    standard_deviation = get_weighted_stdev(tensor, dim=2)
    assert standard_deviation == pytest.approx(0.5791246294975281)


def test_get_weighted_stdev__mask():
    tensor = torch.Tensor([[[0.33333, 0.33333, 0.33334], [0, 0.5, 0.5]],
                           [[0, 0.5, 0.5], [0, 0.5, 0.5]]])
    mask = torch.ByteTensor([[1, 0], [0, 0]])
    standard_deviation = get_weighted_stdev(tensor, dim=2, mask=mask)
    # Population standard deviation for 1,2,3
    assert standard_deviation == pytest.approx(0.81649658093, rel=1.0e-04)


def test_anomaly_detector():
    min_steps = 10
    anomaly_detector = AnomalyDetector(min_steps=min_steps)
    for _ in range(min_steps):
        assert not anomaly_detector.step(1)
    assert anomaly_detector.step(2)


def test_anomaly_detector__type_high():
    min_steps = 10
    anomaly_detector = AnomalyDetector(min_steps=min_steps, type_=AnomalyDetector.TYPE_HIGH)
    for _ in range(min_steps):
        assert not anomaly_detector.step(1)
    assert anomaly_detector.step(2)


def test_anomaly_detector__type_low():
    min_steps = 10
    anomaly_detector = AnomalyDetector(min_steps=min_steps, type_=AnomalyDetector.TYPE_LOW)
    for _ in range(min_steps):
        assert not anomaly_detector.step(1)
    assert not anomaly_detector.step(2)


def test_anomaly_detector__type_both():
    min_steps = 10
    anomaly_detector = AnomalyDetector(min_steps=min_steps, type_=AnomalyDetector.TYPE_BOTH)
    for _ in range(min_steps):
        assert not anomaly_detector.step(1)
    assert anomaly_detector.max_deviation < 1
    assert anomaly_detector.step(2)
    assert anomaly_detector.step(0)
    assert anomaly_detector.step(float('nan'))
    assert anomaly_detector.step(float('inf'))


class MockModel(nn.Module):
    # REFERENCE: http://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)


def test_get_total_parameters():
    assert 62006 == get_total_parameters(MockModel())


def test_flatten_parameters__smoke_test():
    flatten_parameters(MockModel())
    flatten_parameters(nn.LSTM(10, 10))


def test_get_root_path():
    assert (ROOT_PATH / 'requirements.txt').is_file()


def test_parse_hparam_args():
    hparam_args = ['--foo 0.01', '--bar WaveNet', '--moo.foo=1']
    assert parse_hparam_args(hparam_args) == {'foo': 0.01, 'bar': 'WaveNet', 'moo.foo': 1}


def test_load_most_recent_checkpoint():
    checkpoint = Checkpoint.most_recent('tests/_test_data/**/*.pt')
    assert isinstance(checkpoint, Checkpoint)
    assert 'tests/_test_data/step_100.pt' in str(checkpoint.path)
    assert 166257493 == random.randint(1, 2**31)  # Checkpoint set random generator state


def test_load_most_recent_checkpoint_none():
    checkpoint = Checkpoint.most_recent('tests/_test_data/**/*.abc')
    assert checkpoint is None


def test_load_save_checkpoint():
    model = nn.LSTM(10, 10)
    optimizer = Optimizer(
        torch.optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters())))
    checkpoint = Checkpoint(
        directory='tests/_test_data/', model=model, step=1000, optimizer=optimizer)
    filename = checkpoint.save()
    assert filename.is_file()

    # Smoke test
    Checkpoint.from_path(filename)

    # Clean up
    filename.unlink()


@mock.patch('torch.distributed')
def test_accumulated_metrics(mock_distributed):
    mock_distributed.reduce.return_value = None
    mock_distributed.is_initialized.return_value = True
    metrics = AccumulatedMetrics(type_=torch)
    metrics.add_metric('test', torch.tensor([.5]), torch.tensor(3))
    metrics.add_metrics({'test': torch.tensor([.25])}, 2)

    def callable_(key, value):
        assert key == 'test' and value == 0.4

    metrics.log_step_end(callable_)
    assert metrics.get_epoch_metric('test') == 0.4
    metrics.log_epoch_end(callable_)
    metrics.reset()

    called = False

    def not_called():
        nonlocal called
        called = True

    # No new metrics to report
    metrics.log_step_end(not_called)
    metrics.log_epoch_end(not_called)

    assert not called


def test_balance_list():
    balanced = balance_list(['a', 'a', 'b', 'b', 'c'])
    assert len(balanced) == 3
    assert len(set(balanced)) == 3


def test_balance_list_determinism():
    list_ = [(i, 'a' if i % 2 == 0 else 'b') for i in range(99)]
    balanced = balance_list(list_, get_class=lambda i: i[1], random_seed=123)
    assert len(balanced) == 98
    assert balanced[0][0] == 57


def test_balance_list_get_weight():
    list_ = [(1, 'a'), (1, 'a'), (2, 'b')]
    balanced = balance_list(list_, get_class=lambda i: i[1], get_weight=lambda i: i[0])
    assert len(balanced) == 3


def test_split_list():
    dataset = [1, 2, 3, 4, 5]
    splits = (.6, .2, .2)
    assert split_list(dataset, splits) == [[1, 2, 3], [4], [5]]


def test_split_list_rounding():
    dataset = [1]
    splits = (.33, .33, .34)
    assert split_list(dataset, splits) == [[], [], [1]]


@mock.patch('src.utils.np.save')
@mock.patch('src.utils.np.load')
def test_batch_predict_spectrograms(mock_load, mock_save):
    speaker = Speaker('Test Speaker', Gender.FEMALE)
    input_encoder = InputEncoder(['this is a test'], [speaker])
    frame_channels = 128
    model = SpectrogramModel(
        input_encoder.text_encoder.vocab_size,
        input_encoder.speaker_encoder.vocab_size,
        frame_channels=frame_channels)
    data = [TextSpeechRow(text='this is a test', audio_path=None, speaker=speaker)]
    mock_save.return_value = None
    mock_load.return_value = numpy.array([1])

    predictions = batch_predict_spectrograms(
        data=data,
        input_encoder=input_encoder,
        model=model,
        batch_size=1,
        device=torch.device('cpu'),
        aligned=False)
    assert len(predictions) == 1
    assert torch.is_tensor(predictions[0])

    predictions = batch_predict_spectrograms(
        data=data,
        input_encoder=input_encoder,
        model=model,
        batch_size=1,
        device=torch.device('cpu'),
        filenames=['/tmp/tensor.npy'],
        aligned=False)
    assert len(predictions) == 1
    assert '/tmp/tensor.npy' in str(predictions[0].path)


@mock.patch('src.utils.np.save')
@mock.patch('src.utils.np.load')
def test_batch_predict_spectrograms_sorting(mock_load, mock_save):
    speaker = Speaker('Test Speaker', Gender.FEMALE)
    input_encoder = InputEncoder(['this is a test'], [speaker])
    frame_channels = 128
    model = SpectrogramModel(
        input_encoder.text_encoder.vocab_size,
        input_encoder.speaker_encoder.vocab_size,
        frame_channels=frame_channels)
    data = [TextSpeechRow(text='this is a test', audio_path=None, speaker=speaker)] * 20
    filenames = ['/tmp/tensor_%d.npy' % d for d in range(20)]
    mock_save.return_value = None
    mock_load.return_value = numpy.array([1])

    predictions = batch_predict_spectrograms(
        data=data,
        input_encoder=input_encoder,
        model=model,
        batch_size=1,
        device=torch.device('cpu'),
        filenames=filenames,
        aligned=False)
    assert len(filenames) == 20
    # Ensure predictions are sorted in the right order
    for prediction, filename in zip(predictions, filenames):
        assert filename in str(prediction.path)


@mock.patch('src.utils.np.save')
@mock.patch('src.utils.np.load')
def test_batch_predict_spectrograms__aligned(mock_load, mock_save):
    speaker = Speaker('Test Speaker', Gender.FEMALE)
    input_encoder = InputEncoder(['this is a test'], [speaker])
    frame_channels = 128
    model = SpectrogramModel(
        input_encoder.text_encoder.vocab_size,
        input_encoder.speaker_encoder.vocab_size,
        frame_channels=frame_channels)
    spectrogram = torch.rand(10, frame_channels)
    example = TextSpeechRow(
        text='this is a test', audio_path=None, speaker=speaker, spectrogram=spectrogram)
    mock_save.return_value = None
    mock_load.return_value = numpy.array([1])

    predictions = batch_predict_spectrograms(
        data=[example],
        input_encoder=input_encoder,
        model=model,
        batch_size=1,
        device=torch.device('cpu'),
        aligned=True)
    assert len(predictions) == 1
    assert torch.is_tensor(predictions[0])


def test_random_generator_state():
    # TODO: Test `torch.cuda` random as well.
    state = get_random_generator_state()
    randint = random.randint(1, 2**31)
    numpy_randint = numpy.random.randint(1, 2**31)
    torch_randint = int(torch.randint(1, 2**31, (1,)))

    set_random_generator_state(state)
    post_randint = random.randint(1, 2**31)
    post_numpy_randint = numpy.random.randint(1, 2**31)
    post_torch_randint = int(torch.randint(1, 2**31, (1,)))

    assert randint == post_randint
    assert numpy_randint == post_numpy_randint
    assert torch_randint == post_torch_randint


def _align_and_format(tokens, other, **kwargs):
    cost, alignment = align_tokens(tokens, other, **kwargs)
    return format_alignment(tokens, other, alignment)


def test_align_tokens():
    # Basic cases
    assert align_tokens('', '')[0] == 0
    assert align_tokens('a', '')[0] == 1
    assert align_tokens('', 'a')[0] == 1
    assert align_tokens('abc', '')[0] == 3
    assert align_tokens('', 'abc')[0] == 3
    assert align_tokens('', 'abc', window_length=1)[0] == 3

    # Should just add "a" to the beginning.
    assert align_tokens('abc', 'bc', window_length=1)[0] == 1
    assert align_tokens('abc', 'bc', allow_substitution=lambda a, b: False)[0] == 5
    assert (_align_and_format('abc', 'bc') == (
        'a b c',
        '  b c',
    ))

    # Should just add I to the beginning.
    assert align_tokens('islander', 'slander')[0] == 1
    assert align_tokens('islander', 'slander', window_length=1)[0] == 1
    assert (_align_and_format('islander', 'slander') == (
        'i s l a n d e r',
        '  s l a n d e r',
    ))

    # Needs to substitute M by K, T by M and add an A to the end
    assert align_tokens('mart', 'karma')[0] == 3

    # Substitute K by S, E by I and insert G at the end.
    assert align_tokens('kitten', 'sitting')[0] == 3

    # Should add 4 letters FOOT at the beginning.
    assert align_tokens('ball', 'football')[0] == 4

    assert align_tokens('ball', 'football', window_length=1)[0] == 7
    assert (_align_and_format('ball', 'football', window_length=1) == (
        'b a l       l',
        'f o o t b a l',
    ))
    assert align_tokens('ball', 'football', window_length=2)[0] == 6
    assert (_align_and_format('ball', 'football', window_length=2) == (
        'b a         l l',
        'f o o t b a l l',
    ))
    assert align_tokens('ball', 'football', window_length=3)[0] == 4
    assert (_align_and_format('ball', 'football', window_length=3) == (
        '        b a l l',
        'f o o t b a l l',
    ))

    # Should delete 4 letters FOOT at the beginning.
    assert align_tokens('football', 'foot')[0] == 4

    # Needs to substitute the first 5 chars: INTEN by EXECU
    assert align_tokens('intention', 'execution')[0] == 5

    # Subtitution of words
    assert align_tokens(['Hey', 'There'], ['Hey', 'There'])[0] == 0
    assert align_tokens(['Hey', 'There'], ['Hi', 'There'])[0] == 2
    assert align_tokens(['Hey', 'There'], ['Hi', 'The'])[0] == 4

    # Deletion of word
    assert align_tokens(['Hey', 'There', 'You'], ['Hey', ',', 'There'])[0] == 4
    assert (_align_and_format(['Hey', 'There', 'You'], ['Hey', ',', 'There']) == (
        'Hey   There',
        'Hey , There',
    ))

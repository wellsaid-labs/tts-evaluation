from collections import Counter
from collections import namedtuple

import pytest
import torch

from torchnlp.random import fork_rng_wrap

from src.bin.train.spectrogram_model.data_loader import _BalancedSampler
from src.bin.train.spectrogram_model.data_loader import DataLoader
from tests._utils import get_tts_mocks


def test_data_loader():
    mocks = get_tts_mocks(add_spectrogram=True)
    data = mocks['dev_dataset']
    batch_size = 2

    # Smoke test
    iterator = DataLoader(data, batch_size, mocks['device'], input_encoder=mocks['input_encoder'])
    assert len(iterator) == len(data) // batch_size

    # Test collate
    samples = list(iterator)  # The iterator contains some randomness everytime it's sampled.
    assert sum([r.spectrogram_mask.tensor.sum().item() for r in samples]) == (
        sum([r.spectrogram[1].sum().item() for r in samples]))
    assert sum([r.spectrogram_mask.tensor.sum().item() for r in samples]) == (
        sum([r.spectrogram.tensor.sum(dim=2).nonzero().shape[0] for r in samples]))
    assert sum([r.spectrogram_expanded_mask[0].sum().item() for r in samples]) == (
        sum([r.spectrogram.tensor.nonzero().shape[0] for r in samples]))
    assert sum([r.stop_token.tensor.sum().item() for r in samples]) == len(samples) * batch_size


def test__data_loader__expected_average_spectrogram_length():
    """ Given that the data loader loads an equal amount of audio per speaker, check that the
    expected spectrogram length is computed correctly.
    """
    MockExample = namedtuple('MockExample', ('spectrogram', 'speaker'))
    data = [MockExample(torch.zeros(5), 'a'), MockExample(torch.zeros(2), 'b')]
    expected = 5 * (1 / 3.5) + 2 * ((5 / 2) / 3.5)
    iterator = DataLoader(data, 1, torch.device('cpu'))
    assert iterator.expected_average_spectrogram_length.item() == pytest.approx(expected)


@fork_rng_wrap(seed=123)
def test__balanced_sampler():
    """ Ensure that balanced sampler is picking an equal amount of audio per speaker and
    each example afterwards has an equal chance of getting selected.
    """
    # NOTE: For example, 'a' could be a speaker and 1 could be the audio length.
    data = [('a', 1), ('a', 200), ('b', 2), ('c', 1)]
    num_samples = 1000000
    sampler = _BalancedSampler(
        data,
        replacement=True,
        num_samples=num_samples,
        get_class=lambda e: e[0],
        get_weight=lambda e: e[1])
    samples = [data[i] for i in sampler]
    counts = Counter(samples)

    # NOTE: In our example, this is the total audio in the data sample.
    total = sum(counts[r] * r[1] for r in data)

    # NOTE: In our example, this ensures that each speaker has had an equal amount of audio
    # drawn.
    assert (counts[data[0]] * data[0][1] + counts[data[1]] * data[1][1]) / total == pytest.approx(
        .33, rel=1e-1)
    assert (counts[data[2]] * data[2][1]) / total == pytest.approx(.33, rel=1e-1)
    assert (counts[data[3]] * data[3][1]) / total == pytest.approx(.33, rel=1e-1)

    # NOTE: In our example, within a particular speaker, each example is treated equally
    # regardless of audio length. This means that each unit of audio has an equal chance of
    # getting sampled; therefore, this is assuming that any downstream service will also
    # treat each unit of audio equally.
    assert counts[data[0]] == pytest.approx(counts[data[1]], rel=1e-1)

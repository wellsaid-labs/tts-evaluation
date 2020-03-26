from collections import Counter
from unittest import mock

from torchnlp.random import fork_rng
from torchnlp.random import fork_rng_wrap

import pytest
import torch

from src.bin.train.signal_model.data_loader import _get_slice
from src.bin.train.signal_model.data_loader import _BalancedSampler
from src.bin.train.signal_model.data_loader import DataLoader
from tests._utils import get_tts_mocks


@fork_rng_wrap(seed=123)
def test__balanced_sampler():
    """ Ensure that balanced sampler is picking an equal amount of audio per speaker and
    each example is weighted by the audio length.
    """
    # NOTE: For example, 'a' could be a speaker and 1 is the audio length.
    data = [('a', 1), ('a', 200), ('b', 2), ('c', 1), ('d', 1), ('d', 2), ('d', 0)]
    num_samples = 10000000
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
        .25, rel=1e-1)
    assert (counts[data[2]] * data[2][1]) / total == pytest.approx(.25, rel=1e-1)
    assert (counts[data[3]] * data[3][1]) / total == pytest.approx(.25, rel=1e-1)
    assert (counts[data[4]] * data[4][1] + counts[data[5]] * data[5][1]) / total == pytest.approx(
        .25, rel=1e-1)

    # NOTE: In our example, within a particular speaker, each example is weighted by it's length;
    # therefore, `('a', 200)` will be sampled 200x more times than `('a', 1)`.
    assert counts[data[0]] / data[0][1] == pytest.approx(counts[data[1]] / data[1][1], rel=1e-1)
    assert counts[data[4]] / data[4][1] == pytest.approx(counts[data[5]] / data[5][1], rel=1e-1)


@fork_rng_wrap(seed=123)
def test__balanced_sampler__dataset():
    """ Ensure that `_BalancedSampler` works in a real dataset. """
    data = get_tts_mocks(add_spectrogram=True)['dataset']
    num_samples = 10000
    sampler = _BalancedSampler(
        data,
        replacement=True,
        num_samples=num_samples,
        get_class=lambda e: e.speaker,
        get_weight=lambda e: e.spectrogram.shape[0])
    samples = [data[i] for i in sampler]
    get_key = lambda e: (e.speaker, e.text, e.spectrogram.shape[0])
    counts = Counter([get_key(e) for e in samples])

    total_frames = sum(counts[get_key(e)] * e.spectrogram.shape[0] for e in data)
    speakers = set([e.speaker for e in data])

    for speaker in speakers:
        # Ensure that there are an equal number of frames per speaker sampled.
        samples_per_speaker = [e for e in samples if e.speaker == speaker]
        num_frames = sum([e.spectrogram.shape[0] for e in samples_per_speaker])
        assert num_frames / total_frames == pytest.approx(1 / len(speakers), rel=1e-1)

        # Ensure that every example is sampled proportionally to it's size
        counts_per_speaker = Counter([get_key(e) for e in samples_per_speaker])
        total_frames_per_speaker = sum(
            [e.spectrogram.shape[0] for e in data if e.speaker == speaker])
        assert all([
            c / len(samples_per_speaker) == pytest.approx(
                e[2] / total_frames_per_speaker, rel=1e-1) for e, c in counts_per_speaker.items()
        ])


@mock.patch('src.bin.train.signal_model.data_loader.random.randint')
def test__get_slice(randint_mock):
    randint_mock.return_value = 5
    samples_per_frame = 10
    spectrogram_channels = 80
    spectrogram = torch.rand(10, spectrogram_channels)
    signal = torch.rand(100)
    slice_pad = 3
    slice_size = 3
    slice_ = _get_slice(
        spectrogram, signal, spectrogram_slice_size=slice_size, spectrogram_slice_pad=slice_pad)

    assert slice_.spectrogram.shape == (slice_size + slice_pad * 2, spectrogram_channels)
    assert slice_.spectrogram_mask.shape == (slice_size + slice_pad * 2,)
    assert slice_.target_signal.shape == (slice_size * samples_per_frame,)
    assert slice_.signal_mask.shape == (slice_size * samples_per_frame,)


def test__get_slice__distribution():
    """ Test that `_get_slice` samples each sample equally. """
    spectrogram = torch.arange(1, 5).unsqueeze(1)
    signal = torch.arange(1, 13)
    slice_size = 3
    spectrogram_slice_pad = 3
    samples = 10000
    sample_counter = Counter()
    frame_counter = Counter()

    for i in range(samples):
        slice_ = _get_slice(
            spectrogram,
            signal,
            spectrogram_slice_size=slice_size,
            spectrogram_slice_pad=spectrogram_slice_pad)
        sample_counter.update(slice_.target_signal.tolist())
        frame_counter.update(slice_.spectrogram.squeeze().tolist())

    total_samples = sum(sample_counter.values()) - sample_counter[0]  # Remove padding
    for i in range(signal.shape[0]):
        # Each sample should be sampled `1 / signal.shape[0]` times
        assert sample_counter[signal[i].item()] / total_samples == pytest.approx(
            1 / signal.shape[0], rel=1e-1)

    total_frames = sum(frame_counter.values()) - frame_counter[0]  # Remove padding
    for i in range(spectrogram.shape[0]):
        assert frame_counter[spectrogram[i, 0].item()] / total_frames == pytest.approx(
            1 / spectrogram.shape[0], rel=1e-1)


@mock.patch('src.bin.train.signal_model.data_loader.random.randint')
def test__get_slice__padding(randint_mock):
    randint_mock.return_value = 1
    spectrogram = torch.tensor([[1], [2], [3]])
    signal = torch.tensor([.1, .1, .2, .2, .3, .3])

    slice_pad = 3
    slice_size = 2
    slice_ = _get_slice(
        spectrogram, signal, spectrogram_slice_size=slice_size, spectrogram_slice_pad=slice_pad)

    assert torch.equal(slice_.spectrogram, torch.tensor([[0], [0], [1], [2], [3], [0], [0], [0]]))
    assert torch.equal(slice_.spectrogram_mask,
                       torch.tensor([0, 0, 1, 1, 1, 0, 0, 0], dtype=torch.bool))
    assert torch.equal(slice_.target_signal, torch.tensor([0.2, 0.2, 0.3, 0.3]))
    assert torch.equal(slice_.signal_mask, torch.tensor([1, 1, 1, 1], dtype=torch.bool))


def test_data_loader():
    mocks = get_tts_mocks(add_spectrogram=True)
    data = mocks['dev_dataset']
    samples_per_frame = data[0].spectrogram_audio.shape[0] / data[0].spectrogram.shape[0]
    batch_size = 2
    slice_size = 75
    slice_pad = 0

    device = torch.device('cpu')

    # Smoke test
    loader = DataLoader(
        data,
        batch_size,
        device,
        use_predicted=False,
        spectrogram_slice_size=slice_size,
        spectrogram_slice_pad=slice_pad)
    assert len(loader) == (len(data) // batch_size)

    # Ensure that sampled batches are different each time.
    with fork_rng(seed=124):
        samples = list(loader)
        assert len(set([s.spectrogram[0].sum().item() for s in samples])) == len(samples)
        more_samples = list(loader)
        assert len(set([s.spectrogram[0].sum().item() for s in more_samples + samples])) == (
            len(samples) + len(more_samples))

    # Test collate
    item = next(iter(loader))
    assert item.signal_mask.sum() <= slice_size * batch_size * samples_per_frame

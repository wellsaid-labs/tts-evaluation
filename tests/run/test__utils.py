import collections
import shutil
import tempfile
import typing
from pathlib import Path

import hparams
import numpy as np
import pytest
import torch
import torch.nn
import torchnlp
import torchnlp.random
from matplotlib import pyplot

import lib
import run
from lib.audio import AudioFileMetadata
from lib.datasets import Alignment
from run._config import Cadence, DatasetType, get_dataset_label
from tests import _utils

TEST_DATA_PATH = _utils.TEST_DATA_PATH / "audio"
TEST_DATA_LJ = TEST_DATA_PATH / "bit(rate(lj_speech,24000),32).wav"


@pytest.fixture(autouse=True)
def run_around_tests():
    """ Set a basic configuration. """
    sample_rate = 24000
    hparams.add_config({lib.audio.write_audio: hparams.HParams(sample_rate=sample_rate)})
    yield
    hparams.clear_config()


def _make_passage(
    alignments: typing.Tuple[Alignment, ...],
    speaker: lib.datasets.Speaker,
    metadata=_utils.make_metadata(),
) -> lib.datasets.Passage:
    """ Make `Passage` for testing. """
    range_ = lambda a: "".join([str(i) for i in range(a)])
    script = range_(max([a.script[1] for a in alignments])) if len(alignments) != 0 else ""
    transcript = range_(max([a.transcript[1] for a in alignments])) if len(alignments) != 0 else ""
    return lib.datasets.Passage(metadata, speaker, script, transcript, alignments)


def test_maybe_make_experiment_directories(capsys):
    """ Test `maybe_make_experiment_directories` creates a directory structure. """
    with tempfile.TemporaryDirectory() as directory:
        with capsys.disabled():  # NOTE: Disable capsys because it messes with `sys.stdout`
            path = Path(directory)
            recorder = lib.environment.RecordStandardStreams()
            run_name = "run_name"
            checkpoints_directory_name = "checkpoints"
            run_log_filename = "run.log"
            run_root, checkpoints_directory = run._utils.maybe_make_experiment_directories(
                path,
                recorder,
                run_name=run_name,
                checkpoints_directory_name=checkpoints_directory_name,
                run_log_filename=run_log_filename,
            )
            assert run_root.is_dir()
            assert run_root.parent == path
            assert run_root.name == run_name
            assert checkpoints_directory.is_dir()
            assert checkpoints_directory.parent == run_root
            assert checkpoints_directory.name == checkpoints_directory_name
            assert recorder.log_path.parent == run_root
            assert recorder.log_path.name == run_log_filename


def test_maybe_make_experiment_directories_from_checkpoint(capsys):
    """ Test `maybe_make_experiment_directories_from_checkpoint` creates a directory structure. """
    with tempfile.TemporaryDirectory() as directory:
        with capsys.disabled():  # NOTE: Disable capsys because it messes with `sys.stdout`
            path = Path(directory)
            checkpoints_directory = path / "run_name" / "checkpoint_directory_name"
            checkpoints_directory.mkdir(parents=True)
            recorder = lib.environment.RecordStandardStreams()
            run_name = "new_run"
            checkpoint = run._utils.Checkpoint(checkpoints_directory, "", 0)
            run_root, _ = run._utils.maybe_make_experiment_directories_from_checkpoint(
                checkpoint, recorder, run_name=run_name
            )
            assert run_root.parent == path
            assert run_root.name == run_name


def test_normalize_audio():
    """Test `run._utils.normalize_audio` normalizes audio in `dataset`."""
    sample_rate = 8000
    num_channels = 2
    ffmpeg_encoding = "pcm_s16le"
    sox_encoding = "16-bit Signed Integer PCM"
    suffix = ".wav"
    args = (sample_rate, num_channels, sox_encoding, 7.584, "256k", "16-bit")
    with tempfile.TemporaryDirectory() as path:
        directory = Path(path)
        audio_path = directory / TEST_DATA_LJ.name
        shutil.copy(TEST_DATA_LJ, audio_path)
        metadata = AudioFileMetadata(audio_path, *args)
        passage = _make_passage(tuple(), speaker=lib.datasets.LINDA_JOHNSON, metadata=metadata)
        dataset = {lib.datasets.LINDA_JOHNSON: [passage]}
        run._utils.normalize_audio(
            dataset,
            suffix=suffix,
            encoding=ffmpeg_encoding,
            sample_rate=sample_rate,
            num_channels=num_channels,
            audio_filters=lib.audio.AudioFilters(""),
        )
        assert len(dataset[lib.datasets.LINDA_JOHNSON]) == 1
        new_path = dataset[lib.datasets.LINDA_JOHNSON][0].audio_file.path
        assert new_path.absolute() != audio_path.absolute()
        assert lib.audio.get_audio_metadata(new_path) == AudioFileMetadata(new_path, *args)


def test_split_passages():
    """Test `run._utils.split_passages` randomly splits `passages` into train and dev lists. """
    _make = lambda a, s: _make_passage((Alignment((0, a), (0, a), (0, a)),), s)
    with torchnlp.random.fork_rng(123):
        a = lib.datasets.Speaker("a")
        b = lib.datasets.Speaker("b")
        passages = [_make(1, a), _make(2, a), _make(3, a), _make(1, b), _make(2, b), _make(3, b)]
        train, dev = run._utils.split_passages(passages, 6)
        assert train == [_make(3, b), _make(3, a), _make(1, a)]
        assert dev == [_make(1, b), _make(2, b), _make(2, a)]


def test_split_passages__empty_list():
    """Test `run._utils.split_passages` errors if there are not enough passages. """
    with pytest.raises(AssertionError):
        run._utils.split_passages([], 6)


def test__get_normalized_half_gaussian():
    """Test `run._utils._get_normalized_half_gaussian` generates the left-side of a gaussian
    distribution normalized from 0 to 1."""
    _utils.assert_almost_equal(
        run._utils._get_normalized_half_gaussian(8, 2),
        torch.tensor(
            [0.0015184, 0.0070632, 0.0292409, 0.0952311, 0.2443498, 0.4946532, 0.7909854, 1.0]
        ),
    )


def test__random_nonoverlapping_alignments():
    """Test `run._utils._random_nonoverlapping_alignments` generates the left-side of a gaussian
    distribution normalized from 0 to 1."""
    _alignment = lambda a, b: Alignment((a, b), (a, b), (a, b))
    alignments = tuple(
        [
            _alignment(0, 1),
            _alignment(1, 2),
            _alignment(2, 3),
            _alignment(3, 4),
            _alignment(4, 5),
        ]
    )
    counter: typing.Counter[int] = collections.Counter()
    for i in range(100000):
        samples = run._utils._random_nonoverlapping_alignments(alignments, 3)
        for sample in samples:
            for i in range(sample.script[0], sample.script[1]):
                counter[i] += 1
    assert set(counter.keys()) == set(range(0, 5))
    _utils.assert_uniform_distribution(counter, abs=0.02)


def test__random_nonoverlapping_alignments__empty():
    """Test `run._utils._random_nonoverlapping_alignments` handles empty list. """
    assert run._utils._random_nonoverlapping_alignments(tuple(), 3) == tuple()


def test__random_nonoverlapping_alignments__large_max():
    """Test `run._utils._random_nonoverlapping_alignments` handles a large maximum. """
    _alignment = lambda a, b: Alignment((a, b), (a, b), (a, b))
    with torchnlp.random.fork_rng(1234):
        alignments = tuple(
            [
                _alignment(0, 1),
                _alignment(1, 2),
                _alignment(2, 3),
                _alignment(3, 4),
                _alignment(4, 5),
            ]
        )
        assert len(run._utils._random_nonoverlapping_alignments(alignments, 1000000)) == 6


def test_seconds_to_samples():
    """Test `run._utils.seconds_to_samples` handles a basic case."""
    assert run._utils.seconds_to_samples(1.5, 24000) == 36000


def test__get_loudness():
    """Test `run._utils._get_loudnes` slices, measures, and rounds loudness correctly. """
    sample_rate = 1000
    length = 10
    implementation = "K-weighting"
    meter = lib.audio.get_pyloudnorm_meter(sample_rate, implementation)
    with torchnlp.random.fork_rng(12345):
        audio = np.random.rand(sample_rate * length) * 2 - 1  # type: ignore
        alignment = Alignment((0, length), (0, length), (0, length))
        loundess = run._utils._get_loudness(audio, sample_rate, alignment, implementation, 1)
        assert np.isfinite(loundess)  # type: ignore
        assert round(meter.integrated_loudness(audio), 1) == loundess


def test__get_words():
    """ Test `run._utils._get_words` parses a sentence correctly. """
    text = "It was time to present the present abcdefghi."
    character_to_word, word_vectors, word_pronunciations, phonemes = run._utils._get_words(
        text, start=0, stop=len(text), separator="|"
    )
    # fmt: off
    assert character_to_word == [0, 0, -1, 1, 1, 1, -1, 2, 2, 2, 2, -1, 3, 3, -1, 4, 4, 4, 4, 4, 4,
                                 4, -1, 5, 5, 5, -1, 6, 6, 6, 6, 6, 6, 6, -1, 7, 7, 7, 7, 7, 7, 7,
                                 7, 7, 8]
    # fmt: on
    assert word_vectors.shape == (len(text), 300)
    # NOTE: `-1`/`7` and "present"/"present" have the same word vector.
    assert np.unique(word_vectors, axis=0).shape == (  # type: ignore
        len(set(character_to_word)) - 2,
        300,
    )
    assert word_vectors[0].sum() != 0
    slice_ = slice(-11, -1)
    assert character_to_word[slice_] == [-1, 7, 7, 7, 7, 7, 7, 7, 7, 7]
    # NOTE: OOV words and non-word characters should have a zero vectors.
    assert word_vectors[slice_].sum() == 0
    assert word_pronunciations == (
        ("IH1", "T"),
        None,
        ("T", "AY1", "M"),
        None,
        ("P", "R", "IH0", "Z", "EH1", "N", "T"),  # Verb
        None,
        ("P", "R", "EH1", "Z", "AX", "N", "T"),  # Noun
        None,
        None,
    )
    assert phonemes == (
        "ɪ|t| |w|ʌ|z| |t|ˈ|aɪ|m| |t|ə| |p|ɹ|ɪ|z|ˈ|ɛ|n|t| "
        "|ð|ə| |p|ɹ|ˈ|ɛ|z|ə|n|t| |ɐ|b|k|d|ˈ|ɛ|f|ɡ|i|."
    )


def test__get_words__bad_slice():
    """ Test `run._utils._get_words` does not slice words. """

    text = "It was time to present the present abcdefghi."
    assert text[0:-11] == "It was time to present the present"
    run._utils._get_words(text, start=0, stop=-11)
    assert text[0:-10] == "It was time to present the present "
    run._utils._get_words(text, start=0, stop=-10)

    with pytest.raises(AssertionError):
        assert text[0:-12] == "It was time to present the presen"
        run._utils._get_words(text, start=0, stop=-12)
    with pytest.raises(AssertionError):
        assert text[0:-9] == "It was time to present the present a"
        run._utils._get_words(text, start=0, stop=-9)


def test__get_words__slice():
    """ Test `run._utils._get_words` parses a sentence correctly. """
    text = "It was time to present the present abcdefghi."
    character_to_word, word_vectors, word_pronunciations, phonemes = run._utils._get_words(
        text, start=3, stop=len(text), separator="|"
    )
    # fmt: off
    assert character_to_word == [0, 0, 0, -1, 1, 1, 1, 1, -1, 2, 2, -1, 3, 3, 3, 3, 3, 3, 3, -1, 4,
                                 4, 4, -1, 5, 5, 5, 5, 5, 5, 5, -1, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7]
    # fmt: on
    assert word_vectors.shape == (len(text) - 3, 300)
    # NOTE: `-1`/`7` and "present"/"present" have the same word vector.
    assert np.unique(word_vectors, axis=0).shape == (  # type: ignore
        len(set(character_to_word)) - 2,
        300,
    )
    assert word_pronunciations == (
        None,
        ("T", "AY1", "M"),
        None,
        ("P", "R", "IH0", "Z", "EH1", "N", "T"),  # Verb
        None,
        ("P", "R", "EH1", "Z", "AX", "N", "T"),  # Noun
        None,
        None,
    )
    assert phonemes == (
        "w|ʌ|z| |t|ˈ|aɪ|m| |t|ə| |p|ɹ|ɪ|z|ˈ|ɛ|n|t| |ð|ə| |p|ɹ|ˈ|ɛ|z|ə|n|t| |ɐ|b|k|d|ˈ|ɛ|f|ɡ|i|."
    )


def test_worker_init_fn():
    """Test `run._utils.worker_init_fn` sets a deterministic but random seed. """
    run._utils.worker_init_fn(1, 123, 1)
    assert torch.randn(1).item() == -0.9041745066642761
    run._utils.worker_init_fn(1, 123, 1)
    assert torch.randn(1).item() == -0.9041745066642761
    run._utils.worker_init_fn(1, 123, 2)
    assert torch.randn(1).item() == -1.5076009035110474


def test_set_context():
    """Test `run._utils.set_context` updates comet, module, and grad context. """
    comet = run._utils.CometMLExperiment(disabled=True)
    rnn = torch.nn.LSTM(10, 20, 2).eval()
    assert not rnn.training
    with run._utils.set_context(run._utils.Context.TRAIN, rnn, comet):
        assert rnn.training
        assert comet.context == run._utils.Context.TRAIN.value
        output, _ = rnn(torch.randn(5, 3, 10))
        assert output.requires_grad
    assert not rnn.training


def _get_db_spectrogram(signal, **kwargs) -> torch.Tensor:
    spectrogram = torch.stft(signal.view(1, -1), **kwargs)
    spectrogram = torch.norm(spectrogram.double(), dim=-1)
    return lib.audio.amplitude_to_db(spectrogram).permute(2, 0, 1)


def test_get_rms_level():
    """ Test `run._utils.get_rms_level` gets an approximate dB RMS level from a dB spectrogram. """
    frame_length = 1024
    frame_hop = frame_length // 4
    window = torch.ones(frame_length)
    _db_spectrogram = lambda s: _get_db_spectrogram(
        torch.tensor(s),
        n_fft=frame_length,
        hop_length=frame_hop,
        win_length=len(window),
        window=window,
        center=False,
    )
    db_spectrogram = torch.cat(
        [
            _db_spectrogram(lib.audio.full_scale_square_wave()),
            _db_spectrogram(lib.audio.full_scale_sine_wave()),
        ],
        dim=1,
    )
    rms = run._utils.get_rms_level(db_spectrogram, window=window)
    _utils.assert_almost_equal(rms / db_spectrogram.shape[0], torch.Tensor([1.0000001, 0.500006]))


def test_get_rms_level__precise():
    """ Test `run._utils.get_rms_level` gets an exact dB RMS level from a dB spectrogram. """
    frame_length = 1024
    frame_hop = frame_length // 4
    window = torch.ones(frame_length)
    sample_rate = 48000
    _db_spectrogram = lambda s: _get_db_spectrogram(
        lib.utils.pad_tensor(torch.tensor(s), (frame_length, frame_length)),
        n_fft=frame_length,
        hop_length=frame_hop,
        win_length=len(window),
        window=window,
        center=False,
    )
    db_spectrogram = torch.cat(
        [
            _db_spectrogram(lib.audio.full_scale_square_wave()),
            _db_spectrogram(lib.audio.full_scale_sine_wave()),
        ],
        dim=1,
    )
    rms = run._utils.get_rms_level(db_spectrogram, window=window)
    _utils.assert_almost_equal(rms / (sample_rate / frame_hop), torch.Tensor([1.0, 0.49999998418]))


def test_get_dataset_stats():
    """ Test `run._utils.get_dataset_stats` measures dataset statistics correctly. """
    _passage = lambda a, b, s: _make_passage((Alignment((a, b), (a * 10, b * 10), (a, b)),), s)
    a = lib.datasets.Speaker("a")
    b = lib.datasets.Speaker("b")
    train = {a: [_passage(0, 2, a), _passage(0, 2, a)], b: [_passage(0, 1, a)]}
    stats = run._utils.get_dataset_stats(train, {})
    static = Cadence.STATIC
    get_label = lambda n, t, s=None: get_dataset_label(n, cadence=static, type_=t, speaker=s)
    assert stats == {
        get_label("num_passages", DatasetType.TRAIN): 3,
        get_label("num_characters", DatasetType.TRAIN): 5,
        get_label("num_seconds", DatasetType.TRAIN): "50s 0ms",
        get_label("num_passages", DatasetType.TRAIN, a): 2,
        get_label("num_characters", DatasetType.TRAIN, a): 4,
        get_label("num_seconds", DatasetType.TRAIN, a): "40s 0ms",
        get_label("num_passages", DatasetType.TRAIN, b): 1,
        get_label("num_characters", DatasetType.TRAIN, b): 1,
        get_label("num_seconds", DatasetType.TRAIN, b): "10s 0ms",
        get_label("num_passages", DatasetType.DEV): 0,
        get_label("num_characters", DatasetType.DEV): 0,
        get_label("num_seconds", DatasetType.DEV): "0ms",
    }


def test_get_num_skipped():
    """ Test `run._utils.get_num_skipped` counts skipped tokens correctly. """
    alignments_ = [
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]],  # Test no skips
        [[1, 0, 0], [0, 1, 0], [0, 1, 0]],  # Test skipped
        [[1, 0, 0], [0, 1, 0], [0, 1, 0]],
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    ]
    alignments = torch.tensor(alignments_).transpose(0, 1).float()
    spectrogram_mask_ = [
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 0],  # Test that a masked frame is ignored
    ]
    spectrogram_mask = torch.tensor(spectrogram_mask_).transpose(0, 1).bool()
    token_mask_ = [
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 0],  # Test that a masked token cannot be skipped
        [1, 1, 1],
    ]
    token_mask = torch.tensor(token_mask_).bool()
    num_skips = run._utils.get_num_skipped(alignments, token_mask, spectrogram_mask)
    assert num_skips.tolist() == [0.0, 1.0, 0.0, 1.0]


def test_comet_ml_experiment():
    """Test if `run._utils.CometMLExperimentw` initializes, and the patched functions execute."""
    comet = run._utils.CometMLExperiment(disabled=True)
    with comet.context_manager(run._utils.Context.TRAIN):
        assert comet.context == str(run._utils.Context.TRAIN)
        comet.set_step(None)
        comet.set_step(0)
        comet.set_step(0)
        comet.set_step(1)
        comet.log_html_audio(
            metadata="random metadata",
            audio={"predicted_audio": torch.rand(100), "gold_audio": torch.rand(100)},
        )
        figure = pyplot.figure()
        pyplot.close(figure)
        comet.log_figures({run._config.Label("figure"): figure})
        comet.log_current_epoch(0)
        comet.log_epoch_end(0)
        comet.set_name("name")
        comet.add_tags(["tag"])

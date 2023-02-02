import logging
import math
import threading

import config as cf
import pytest
from torchnlp.random import fork_rng

import lib
from lib.text import XMLType
from run._models.spectrogram_model import Inputs, preprocess
from run._tts import tts_ffmpeg_generator
from tests.run._utils import make_mock_tts_package

logger = logging.getLogger(__name__)


@pytest.fixture(autouse=True)
def run_around_test():
    yield
    cf.purge()


def _make_args():
    """Create arguments for the below tests."""
    dataset, package = make_mock_tts_package()
    nlp = lib.text.load_en_core_web_sm()
    passage = list(dataset.values())[0][-1]
    script, session = passage.script, passage.session
    # NOTE: The script needs to be long enough to pass the below tests.
    script = " ".join([script] * 3)
    inputs = Inputs.from_xml(XMLType(script), nlp(script), session)
    # NOTE: `4.6875` is an old magic number that this test case was built on, these test cases
    # can be recalibrated to use a different number.
    preprocessed = preprocess(inputs, {}, {}, lambda t: int(len(t) * 4.6875))
    package.spec_model.allow_unk_on_eval(True)
    package.signal_model.allow_unk_on_eval(True)
    return package, inputs, preprocessed


def test_tts_ffmpeg_generator():
    """Test `tts_ffmpeg_generator` generates an MP3 file.

    TODO: Consider configuring the signal and spectrogram model to be smaller and faster.
    """
    with fork_rng(seed=123):
        package, inputs, preprocessed_inputs = _make_args()
        generator = tts_ffmpeg_generator(package, inputs, preprocessed_inputs, **cf.get())
        assert len(b"".join([s for s in generator])) == 416685


def test_tts_ffmpeg_generator__thread_leak():
    """Test `tts_ffmpeg_generator` cleans up threads on generator close."""
    with fork_rng(seed=123):
        package, inputs, preprocessed_inputs = _make_args()
        active_threads = threading.active_count()
        generator = tts_ffmpeg_generator(package, inputs, preprocessed_inputs, **cf.get())
        package.spec_model.stop_threshold = math.inf
        next(generator)
        assert active_threads + 1 == threading.active_count()
        generator.close()
        assert active_threads == threading.active_count()

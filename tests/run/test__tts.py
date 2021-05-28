import logging
import math
import threading

from torchnlp.random import fork_rng

import lib
from run._tts import encode_tts_inputs, text_to_speech_ffmpeg_generator
from tests import _utils

logger = logging.getLogger(__name__)


def _make_args():
    """Create arguments for the below tests."""
    dataset, package = _utils.make_mock_tts_package()
    nlp = lib.text.load_en_core_web_sm(disable=("parser", "ner"))
    passage = list(dataset.values())[0][-1]
    script, speaker, session = passage.script, passage.speaker, passage.session
    # NOTE: The script needs to be long enough to pass the below tests.
    script = " ".join([script] * 3)
    encoded = encode_tts_inputs(nlp, package.input_encoder, script, speaker, session)
    return package, encoded


def test_text_to_speech_ffmpeg_generator():
    """Test `text_to_speech_ffmpeg_generator` generates an MP3 file.

    TODO: Consider configuring the signal and spectrogram model to be smaller and faster.
    """
    with fork_rng(seed=123):
        package, encoded = _make_args()
        generator = text_to_speech_ffmpeg_generator(logger, package, encoded)
        assert len(b"".join([s for s in generator])) == 12045


def test_text_to_speech_ffmpeg_generator__thread_leak():
    """Test `text_to_speech_ffmpeg_generator` cleans up threads on generator close."""
    with fork_rng(seed=123):
        package, encoded = _make_args()
        active_threads = threading.active_count()
        generator = text_to_speech_ffmpeg_generator(logger, package, encoded)
        package.spectrogram_model.stop_threshold = math.inf
        next(generator)
        assert active_threads + 1 == threading.active_count()
        generator.close()
        assert active_threads == threading.active_count()

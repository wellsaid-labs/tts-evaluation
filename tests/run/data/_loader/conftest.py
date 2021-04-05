import dataclasses
import pathlib
import tempfile

import hparams
import pytest
from hparams import HParams

import lib
from run.data import _loader


@pytest.fixture(autouse=True)
def run_around_tests():
    """ Set a basic configuration. """
    suffix = ".wav"
    data_type = lib.audio.AudioDataType.FLOATING_POINT
    bits = 32
    format_ = lib.audio.AudioFormat(
        sample_rate=24000,
        num_channels=1,
        encoding=lib.audio.AudioEncoding.PCM_FLOAT_32_BIT,
        bit_rate="768k",
        precision="25-bit",
    )
    non_speech_segment_frame_length = 50
    temp_dir = tempfile.TemporaryDirectory()
    temp_dir_path = pathlib.Path(temp_dir.name)
    config = {
        _loader.utils.normalize_audio_suffix: HParams(suffix=suffix),
        _loader.utils.normalize_audio: HParams(
            suffix=suffix,
            data_type=data_type,
            bits=bits,
            sample_rate=format_.sample_rate,
            num_channels=format_.num_channels,
        ),
        _loader.utils.is_normalized_audio_file: HParams(audio_format=format_, suffix=suffix),
        _loader.utils._cache_path: HParams(cache_dir=temp_dir_path),
        # NOTE: `get_non_speech_segments` parameters are set based on `vad_workbook.py`. They
        # are applicable to most datasets with little to no noise.
        _loader.utils.get_non_speech_segments_and_cache: HParams(
            low_cut=300, frame_length=non_speech_segment_frame_length, hop_length=5, threshold=-60
        ),
        _loader.data_structures._make_speech_segments: HParams(
            pad=(non_speech_segment_frame_length / 2) / 1000
        ),
        _loader.utils.maybe_normalize_audio_and_cache: HParams(
            suffix=suffix,
            data_type=data_type,
            bits=bits,
            **{f.name: getattr(format_, f.name) for f in dataclasses.fields(format_)},
        ),
    }
    hparams.add_config(config)
    yield
    hparams.clear_config()

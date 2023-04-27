import pathlib
import tempfile

import config as cf
import pytest

import lib
import run
from run.data import _loader


@pytest.fixture(autouse=True, scope="function")
def run_around_tests():
    """Set a basic configuration."""
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
    config = {
        run._config.data._get_loudness_annotation: cf.Args(
            block_size=0.400, bucket_size=1.0, filter_class="DeMan", sample_rate=format_.sample_rate
        ),
        run._config.data._get_tempo_annotation: cf.Args(bucket_size=0.05),
    }
    cf.add(config)

    temp_dir = tempfile.TemporaryDirectory()
    temp_dir_path = pathlib.Path(temp_dir.name).absolute()
    config = {
        _loader.utils.normalize_audio_suffix: cf.Args(suffix=suffix),
        _loader.utils.normalize_audio: cf.Args(
            suffix=suffix,
            data_type=data_type,
            bits=bits,
            sample_rate=format_.sample_rate,
            num_channels=format_.num_channels,
        ),
        _loader.utils.is_normalized_audio_file: cf.Args(audio_format=format_, suffix=suffix),
        _loader.utils._cache_path: cf.Args(cache_dir=temp_dir_path),
        # NOTE: `get_non_speech_segments` parameters are set based on `vad_workbook.py`. They
        # are applicable to most datasets with little to no noise.
        _loader.utils.get_non_speech_segments_and_cache: cf.Args(
            low_cut=300, frame_length=non_speech_segment_frame_length, hop_length=5, threshold=-60
        ),
        _loader.structures._make_speech_segments_helper: cf.Args(
            pad=(non_speech_segment_frame_length / 2) / 1000
        ),
        _loader.utils.maybe_normalize_audio_and_cache: cf.Args(
            suffix=suffix, data_type=data_type, bits=bits, format_=format_
        ),
        _loader.structures._process_sessions: cf.Args(
            get_loudness=cf.partial(run._config.data._get_loudness_annotation),
            get_tempo=cf.partial(run._config.data._get_tempo_annotation),
            cache_dir=temp_dir_path,
        ),
    }
    cf.add(config)
    yield
    cf.purge()

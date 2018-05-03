import argparse
import mock
import os

import numpy as np
import tensorflow as tf

from tensorflow.contrib.framework.python.ops import audio_ops

from src.spectrogram import _read_audio
from src.spectrogram import wav_to_log_mel_spectrogram
from src.spectrogram import log_mel_spectrogram_to_wav
from src.spectrogram import command_line_interface


def test_librosa_tf_decode_wav():
    """ Librosa provides a more flexible API for decoding WAVs. To ensure consistency with TF, we
    test the output is the same.
    """
    wav_filename = 'tests/_test_data/lj_speech.wav'

    audio_binary = tf.read_file(wav_filename)
    tf_audio, _ = audio_ops.decode_wav(audio_binary)

    audio, _ = _read_audio(wav_filename, sample_rate=None)

    np.testing.assert_array_equal(tf_audio, audio)


@mock.patch(
    'argparse.ArgumentParser.parse_args',
    return_value=argparse.Namespace(path='tests/_test_data/lj_speech.wav'))
def test_command_line_interface(_):
    command_line_interface()
    assert os.path.isfile('tests/_test_data/lj_speech_spectrogram.png')

    # Clean up
    os.remove('tests/_test_data/lj_speech_spectrogram.png')


@mock.patch(
    'argparse.ArgumentParser.parse_args', return_value=argparse.Namespace(path='tests/_test_data/'))
def test_command_line_interface_multiple_files(_):
    command_line_interface()
    assert os.path.isfile('tests/_test_data/lj_speech_spectrogram.png')
    assert os.path.isfile('tests/_test_data/voice_over_spectrogram.png')

    # Clean up
    os.remove('tests/_test_data/lj_speech_spectrogram.png')
    os.remove('tests/_test_data/voice_over_spectrogram.png')


def test_wav_to_log_mel_spectrogram_smoke():
    """ Smoke test to ensure everything runs.
    """
    wav_filename = 'tests/_test_data/lj_speech.wav'
    log_mel_spectrogram = wav_to_log_mel_spectrogram(wav_filename)

    assert log_mel_spectrogram.shape == (603, 80)


def test_log_mel_spectrogram_to_wav_smoke():
    """ Smoke test to ensure everything runs.
    """
    wav_filename = 'tests/_test_data/lj_speech.wav'
    new_wav_filename = 'tests/_test_data/lj_speech_reconstructed.wav'
    log_mel_spectrogram = wav_to_log_mel_spectrogram(wav_filename)
    log_mel_spectrogram_to_wav(log_mel_spectrogram, new_wav_filename, log=True)

    assert os.path.isfile(new_wav_filename)

    # Clean up
    os.remove(new_wav_filename)

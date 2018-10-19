from pathlib import Path

import numpy as np

from src.audio import read_audio
from src.audio import get_log_mel_spectrogram
from src.audio import griffin_lim


def test_log_mel_spectrogram_smoke():
    """ Smoke test to ensure everything runs.
    """
    frame_size = 1200
    frame_hop = 300
    audio_filename = 'tests/_test_data/lj_speech_24000.wav'
    sample_rate = 24000
    signal = read_audio(audio_filename, sample_rate)
    log_mel_spectrogram, padding = get_log_mel_spectrogram(
        signal, sample_rate, frame_size=frame_size, frame_hop=frame_hop)

    assert log_mel_spectrogram.dtype == np.float32
    assert len(log_mel_spectrogram.shape) == 2
    assert len(signal.shape) == 1
    assert int(signal.shape[0] + sum(padding)) / int(log_mel_spectrogram.shape[0]) == frame_hop


def test_griffin_lim_smoke():
    """ Smoke test to ensure everything runs.
    """
    audio_filename = Path('tests/_test_data/lj_speech_24000.wav')
    sample_rate = 24000
    signal = read_audio(audio_filename, sample_rate)
    log_mel_spectrogram, _ = get_log_mel_spectrogram(signal, sample_rate)
    waveform = griffin_lim(log_mel_spectrogram, sample_rate)
    assert len(waveform) > 0

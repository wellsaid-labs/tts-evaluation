import functools

from tensorflow.contrib.signal.python.ops import window_ops

from src.configurable import add_config
from src.configurable import log_config


def set_hparams():
    add_config({
        'src.spectrogram': {
            # SOURCE (Tacotron 1):
            # We use 24 kHz sampling rate for all experiments.
            'read_audio.sample_rate': 24000,
            # SOURCE (Tacotron 2):
            # "mel spectrograms are computed through a shorttime Fourier transform (STFT) using a 50
            # ms frame size, 12.5 ms frame hop, and a Hann window function."
            'wav_to_log_mel_spectrograms': {
                # SOURCE (Tacotron 2):
                # mel spectrograms are computed through a shorttime Fourier transform (STFT) using a
                # 50 ms frame size, 12.5 ms frame hop, and a Hann window function.
                'frame_size': 50,
                'frame_hop': 12.5,
                'window_function': functools.partial(window_ops.hann_window, periodic=True),
                # SOURCE (Tacotron 2):
                # We transform the STFT magnitude to the mel scale using an 80 channel mel
                # filterbank spanning 125 Hz to 7.6 kHz, followed by log dynamic range compression.
                'num_mel_bins': 80,
                'lower_hertz': 125,
                'upper_hertz': 7500,
                # SOURCE (Tacotron 2):
                # Prior to log compression, the filterbank output magnitudes are clipped to a
                # minimum value of 0.01 in order to limit dynamic range in the logarithmic domain.
                'min_magnitude': 0.01,
            }
        }
    })
    log_config()

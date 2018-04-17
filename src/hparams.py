spectrogram = {
    # SOURCE:
    # "mel spectrograms are computed through a shorttime Fourier transform (STFT) using a 50 ms
    # frame size, 12.5 ms frame hop, and a Hann window function."
    'frame': {
        'size': 50,  # milliseconds
        'hop': 12.5,  # milliseconds
    },
    'window_function': 'hann',  # Hann window function
    # SOURCE:
    # We transform the STFT magnitude to the mel scale using an 80 channel mel filterbank spanning
    # 125 Hz to 7.6 kHz, followed by log dynamic range compression.
    'mel': {
        'filterbank': {
            'min': 125,  # Hz
            'max': 7500  # Hz or 7.5 kHz
        },
        # SOURCE:
        # Prior to log compression, the filterbank output magnitudes are clipped to a minimum value
        # of 0.01 in order to limit dynamic range in the logarithmic domain.
        'clip': 0.01,
    }
}

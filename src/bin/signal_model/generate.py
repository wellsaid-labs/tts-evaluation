# python3 -m pip install lws==1.0

import matplotlib
matplotlib.use('Agg')

import os
import random
import logging

import librosa
import librosa.filters
import lws
import numpy as np
from tqdm import tqdm

from src.audio import find_silence
from src.audio import mu_law_quantize
from src.audio import read_audio
from src.datasets import lj_speech_dataset
from src.utils import split_dataset
from src.hparams import set_hparams

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
_mel_basis = None
hop_size = 300
sample_rate = 24000
fft_size = 1200
num_mels = 80
min_level_db = -100
ref_level_db = 20
silence_threshold = 2
rescaling_max = 0.999


def _linear_to_mel(spectrogram):
    global _mel_basis
    if _mel_basis is None:
        _mel_basis = _build_mel_basis()
    return np.dot(_mel_basis, spectrogram)


def _build_mel_basis():
    return librosa.filters.mel(sample_rate, fft_size, n_mels=num_mels)


def _amp_to_db(x):
    return 20 * np.log10(np.maximum(1e-5, x))


def _db_to_amp(x):
    return np.power(10.0, x * 0.05)


def _normalize(S):
    return np.clip((S - min_level_db) / -min_level_db, 0, 1)


def _denormalize(S):
    return (np.clip(S, 0, 1) * -min_level_db) + min_level_db


def _lws_processor():
    return lws.lws(fft_size, hop_size, mode="speech")


def lws_num_frames(length, fsize, fshift):
    """Compute number of time frames of lws spectrogram
    """
    pad = (fsize - fshift)
    if length % fshift == 0:
        M = (length + pad * 2 - fsize) // fshift + 1
    else:
        M = (length + pad * 2 - fsize) // fshift + 2
    return M


def lws_pad_lr(x, fsize, fshift):
    """Compute left and right padding lws internally uses
    """
    M = lws_num_frames(len(x), fsize, fshift)
    pad = (fsize - fshift)
    T = len(x) + 2 * pad
    r = (M - 1) * fshift + fsize - T
    return pad, pad + r


def melspectrogram(y):
    D = _lws_processor().stft(y).T
    S = _amp_to_db(_linear_to_mel(np.abs(D)))
    return _normalize(S)


def _process_utterance(index, wav_path, destination):
    # Load the audio to a numpy array:
    wav, _ = read_audio(wav_path, normalize=False, sample_rate=sample_rate)

    wav = wav / np.abs(wav).max() * rescaling_max

    # Mu-law quantize
    quantized = mu_law_quantize(wav)

    # Trim silences
    start, end = find_silence(quantized, silence_threshold)
    quantized = quantized[start:end]
    wav = wav[start:end]

    # Compute a mel-scale spectrogram from the trimmed wav:
    # (N, D)
    mel_spectrogram = melspectrogram(wav).astype(np.float32).T
    # lws pads zeros internally before performing stft
    # this is needed to adjast time resolution between audio and mel-spectrogram
    l, r = lws_pad_lr(wav, fft_size, hop_size)

    # zero pad for quantized signal
    quantized = np.pad(quantized, (l, r), mode="constant", constant_values=mu_law_quantize(0))
    N = mel_spectrogram.shape[0]
    assert len(quantized) >= N * hop_size
    # time resolution adjastment
    # ensure length of raw audio is multiple of hop_size so that we can use
    # transposed convolution to upsample
    quantized = quantized[:N * hop_size]
    assert quantized.shape[0] % hop_size == 0
    assert len(mel_spectrogram.shape) == 2
    assert len(quantized.shape) == 1
    assert mel_spectrogram.shape[1] == num_mels

    # Write the spectrograms to disk:
    audio_filename = 'quantized_signal_%05d.npy' % index
    mel_filename = 'log_mel_spectrogram_%05d.npy' % index
    np.save(
        os.path.join(destination, audio_filename), quantized.astype(np.int16), allow_pickle=False)
    np.save(
        os.path.join(destination, mel_filename),
        mel_spectrogram.astype(np.float32),
        allow_pickle=False)


def main():
    set_hparams()
    data = lj_speech_dataset()
    random.shuffle(data)
    logger.info('Sample Data:\n%s', data[:5])
    train, dev = split_dataset(data, (0.8, 0.2))
    destination_train = 'data/signal_dataset/train'
    destination_dev = 'data/signal_dataset/dev'

    for (dataset, destination) in [(dev, destination_dev), (train, destination_train)]:
        if not os.path.isdir(destination):
            os.makedirs(destination)

        for i, row in enumerate(tqdm(dataset)):
            _process_utterance(i, row['wav'], destination)


if __name__ == '__main__':  # pragma: no cover
    main()

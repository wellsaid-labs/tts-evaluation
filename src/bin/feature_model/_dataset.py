import logging

from torch.utils import data

import torch
import numpy as np
import librosa

from src.audio import read_audio
from src.audio import get_log_mel_spectrogram
from src.utils.configurable import configurable

logger = logging.getLogger(__name__)


class FeatureDataset(data.Dataset):
    """ Feature dataset loads and preprocesses a signal and text.

    Args:
        dataset (list of dict): List of examples ``dict`` with a ``text (str)`` and ``wav (str)``
            keys where ``wav`` is a filename.
        text_encoder (torchnlp.TextEncoder): Text encoder used to encode and decode the text.
        sample_rate (int): Sample rate of the signal.
    """

    @configurable
    def __init__(self, dataset, text_encoder, sample_rate=24000):
        self.dataset = dataset
        self.text_encoder = text_encoder
        self.sample_rate = sample_rate

    def __len__(self):
        return len(self.dataset)

    def _preprocess_audio(self, wav_filename):
        """ Preprocess a audio: compute spectrogram, pad signal and make stop token targets.

        Args:
            wav_filename (str): Audio file to process.

        Returns:
            log_mel_spectrogram (torch.FloatTensor [num_frames, channels]): Log mel spectrogram
                computed from the signal.
            padded_signal (torch.FloatTensor [signal_length,]): Zero padded signal such that
                ``log_mel_spectrogram.shape[0] % signal.shape[0] == frame_hop``
            stop_token (torch.FloatTensor [num_frames,]): Stop token targets such that
                ``[:-1]`` is composed of zeros and [-1] is a one.
        """
        signal = read_audio(wav_filename, sample_rate=self.sample_rate)
        signal = librosa.effects.trim(signal)[0]

        # Trim silence
        log_mel_spectrogram, padding = get_log_mel_spectrogram(signal, sample_rate=self.sample_rate)
        log_mel_spectrogram = torch.from_numpy(log_mel_spectrogram)
        stop_token = log_mel_spectrogram.new_zeros((log_mel_spectrogram.shape[0],))
        stop_token[-1] = 1

        # Pad so: ``log_mel_spectrogram.shape[0] % signal.shape[0] == frame_hop``
        # We property is required for Wavenet.
        padded_signal = np.pad(signal, padding, mode='constant', constant_values=0)
        padded_signal = torch.from_numpy(padded_signal)
        return log_mel_spectrogram, padded_signal, stop_token

    def __getitem__(self, index):
        example = self.dataset[index]
        log_mel_spectrogram, signal, stop_token = self._preprocess_audio(example['wav'])
        text = self.text_encoder.encode(example['text'])
        return {
            'log_mel_spectrogram': log_mel_spectrogram,
            'signal': signal,
            'stop_token': stop_token,
            'text': text,
        }

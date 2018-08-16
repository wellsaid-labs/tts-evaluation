from multiprocessing import Pool

import logging

from torch.utils import data
from torchnlp.text_encoders import CharacterEncoder

import numpy as np
import torch
import tqdm

from src.utils import get_filename_table

logger = logging.getLogger(__name__)


def read_file(filename):
    """ Return the contents of ``filename``

    Args:
        filename (str)

    Returns:
        (str) contents of filename
    """
    with open(filename, 'r') as file_:
        return file_.read().strip()


def get_spectrogram_length(filename):
    """ Get length of spectrogram (shape [num_frames, num_channels]) from a ``.npy`` numpy file

    Args:
        filename (str): Numpy file

    Returns:
        (int) Length of spectrogram
    """
    return np.load(filename).shape[0]


class FeatureDataset(data.Dataset):
    """ Feature dataset loads and preprocesses a signal and text.

    Args:
        source (str): Directory with data.
        text_encoder (torchnlp.TextEncoder, optional): Text encoder used to encode and decode the
            text.
        load_signal (bool, optional): If ``True``, return signal during iteration.
        log_mel_spectrogram_prefix (str, optional): Prefix of log mel spectrogram files.
        signal_prefix (str, optional): Prefix of signal files.
        text_prefiex (str, optional): Prefix of text files.
    """

    def __init__(self,
                 source,
                 text_encoder=None,
                 load_signal=False,
                 log_mel_spectrogram_prefix='log_mel_spectrogram',
                 signal_prefix='padded_signal',
                 text_prefiex='text'):
        self.spectrogram_key = log_mel_spectrogram_prefix
        self.signal_key = signal_prefix
        self.text_key = text_prefiex
        prefixes = [log_mel_spectrogram_prefix, signal_prefix, text_prefiex]
        self.rows = get_filename_table(source, prefixes=prefixes)

        # Create text_encoder
        if text_encoder is None:
            logger.info('Computing text encoder from %s', source)
            with Pool() as pool:
                filenames = [row[self.text_key] for row in self.rows]
                texts = list(tqdm.tqdm(pool.imap(read_file, filenames), total=len(filenames)))
            self.text_encoder = CharacterEncoder(texts)
        else:
            self.text_encoder = text_encoder

        # Spectrograms lengths for sorting
        logger.info('Computing spectrogram lengths from %s', source)
        with Pool() as pool:
            filenames = [row[self.spectrogram_key] for row in self.rows]
            self.spectrogram_lengths = list(
                tqdm.tqdm(pool.imap(get_spectrogram_length, filenames), total=len(filenames)))

        self.load_signal = load_signal

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        example = self.rows[index]
        log_mel_spectrogram = torch.from_numpy(np.load(example[self.spectrogram_key]))
        signal = torch.from_numpy(np.load(example[self.signal_key])) if self.load_signal else None
        with open(example[self.text_key], 'r') as file_:
            text = file_.read().strip()
        text = self.text_encoder.encode(text)
        stop_token = log_mel_spectrogram.new_zeros((log_mel_spectrogram.shape[0],))
        stop_token[-1] = 1
        return {
            'log_mel_spectrogram': log_mel_spectrogram,
            'signal': signal,
            'stop_token': stop_token,
            'text': text,
        }

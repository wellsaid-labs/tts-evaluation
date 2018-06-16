import logging
import os
import pprint

from tqdm import tqdm

import numpy as np
import librosa

from src.audio import get_log_mel_spectrogram
from src.audio import read_audio
from src.bin.feature_model._utils import set_hparams
from src.datasets import lj_speech_dataset

pretty_printer = pprint.PrettyPrinter(indent=4)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main(destination_train='data/feature_dataset/train',
         destination_dev='data/feature_dataset/dev'):  # pragma: no cover
    """ Main module used to preprocess the signal and spectrogram for training a feature model.

    Args:
        destination_train (str, optional): Directory to save generated files to be used for
            training.
        destination_dev (str, optional): Directory to save generated files to be used for
            development.
    """
    set_hparams()

    if not os.path.isdir(destination_train):
        os.makedirs(destination_train)

    if not os.path.isdir(destination_dev):
        os.makedirs(destination_dev)

    train, dev = lj_speech_dataset()
    logger.info('Sample Data:\n%s', pretty_printer.pformat(train[:5]))
    for dataset, destination in [(train, destination_train), (dev, destination_dev)]:
        for i, row in enumerate(tqdm(dataset)):
            signal = read_audio(row['wav_filename'])

            # Trim silence
            signal = librosa.effects.trim(signal)[0]

            log_mel_spectrogram, padding = get_log_mel_spectrogram(signal)

            # Pad so: ``log_mel_spectrogram.shape[0] % signal.shape[0] == frame_hop``
            # We property is required for Wavenet.
            padded_signal = np.pad(signal, padding, mode='constant', constant_values=0)
            np.save(
                os.path.join(destination, 'padded_signal_%d.npy' % (i,)),
                padded_signal,
                allow_pickle=False)
            np.save(
                os.path.join(destination, 'log_mel_spectrogram_%d.npy' % (i,)),
                log_mel_spectrogram,
                allow_pickle=False)
            with open(os.path.join(destination, 'text_%d.txt' % (i,)), 'w') as file_:
                file_.write(row['text'])


if __name__ == '__main__':
    main()

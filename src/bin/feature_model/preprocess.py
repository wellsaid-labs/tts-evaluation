from pathlib import Path

import logging
import pprint

from torch.multiprocessing import Pool

import numpy as np
import librosa
import tqdm

from src.audio import get_log_mel_spectrogram
from src.audio import read_audio
from src.bin.feature_model._utils import set_hparams
from src.utils.configurable import configurable

pretty_printer = pprint.PrettyPrinter(indent=4)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def process(args):  # pragma: no cover
    index, row, destination = args
    signal = read_audio(row['wav_filename'])

    # Trim silence
    signal = librosa.effects.trim(signal)[0]

    log_mel_spectrogram, padding = get_log_mel_spectrogram(signal)

    # Pad so: ``log_mel_spectrogram.shape[0] % signal.shape[0] == frame_hop``
    # We property is required for the vocoder.
    padded_signal = np.pad(signal, padding, mode='constant', constant_values=0)
    np.save(str(destination / ('padded_signal_%d.npy' % index)), padded_signal, allow_pickle=False)
    np.save(
        str(destination / ('log_mel_spectrogram_%d.npy' % index)),
        log_mel_spectrogram,
        allow_pickle=False)
    (destination / ('text_%d.txt' % index)).write_text(row['text'])


@configurable
def main(dataset,
         destination_train='data/.feature_dataset/train',
         destination_dev='data/.feature_dataset/dev'):  # pragma: no cover
    """ Main module used to preprocess the signal and spectrogram for training a feature model.

    Args:
        dataset (callable, optional): Loads a dataset with train and dev. Each example has
            a ``wav_filename`` and ``text`` key.
        destination_train (str, optional): Directory to save generated files to be used for
            training.
        destination_dev (str, optional): Directory to save generated files to be used for
            development.
    """
    destination_train = Path(destination_train)
    destination_dev = Path(destination_dev)

    if not destination_train.is_dir():
        destination_train.mkdir(parents=True)

    if not destination_dev.is_dir():
        destination_dev.mkdir(parents=True)

    train, dev = dataset()
    logger.info('Sample Data:\n%s', pretty_printer.pformat(train[:5]))
    for dataset, destination in [(train, destination_train), (dev, destination_dev)]:
        args = [(index, row, destination) for index, row in enumerate(dataset)]
        with Pool() as pool:
            list(tqdm.tqdm(pool.imap(process, args), total=len(args)))


if __name__ == '__main__':  # pragma: no cover
    set_hparams()
    main()

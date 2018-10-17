from pathlib import Path

import logging
import pprint
import sys

from torch.multiprocessing import Pool

import numpy as np
import librosa
import tqdm

from src.audio import get_log_mel_spectrogram
from src.audio import read_audio
from src.bin.train.feature_model._utils import set_hparams
from src.utils import duplicate_stream
from src.utils.configurable import configurable
from src.utils.configurable import log_config

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
         destination='data/.feature_dataset/',
         destination_train='train',
         destination_dev='dev',
         destination_stdout='stdout.log',
         destination_stderr='stderr.log'):  # pragma: no cover
    """ Main module used to preprocess the signal and spectrogram for training a feature model.

    Args:
        dataset (callable, optional): Loads a dataset with train and dev. Each example has
            a ``wav_filename`` and ``text`` key.
        destination_train (str, optional): Directory to save generated files to be used for
            training.
        destination_dev (str, optional): Directory to save generated files to be used for
            development.
        destination_stdout (str, optional): Filename to save stderr logs in.
        destination_stderr (str, optional): Filename to save stdout logs in.
    """
    destination = Path(destination)
    destination_train = destination / destination_train
    destination_train.mkdir(parents=True, exist_ok=True)

    destination_dev = destination / destination_dev
    destination_dev.mkdir(parents=True, exist_ok=True)

    duplicate_stream(sys.stdout, destination / destination_stdout)
    duplicate_stream(sys.stderr, destination / destination_stderr)

    log_config()

    train, dev = dataset()
    logger.info('Sample Data:\n%s', pretty_printer.pformat(train[:5]))
    for dataset, destination in [(train, destination_train), (dev, destination_dev)]:
        args = [(index, row, destination) for index, row in enumerate(dataset)]
        with Pool() as pool:
            list(tqdm.tqdm(pool.imap(process, args), total=len(args)))


if __name__ == '__main__':  # pragma: no cover
    set_hparams()
    main()

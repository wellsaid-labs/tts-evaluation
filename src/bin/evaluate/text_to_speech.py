"""
Generate random samples of text to speech model to evaluate.

Example:

    python3 -m src.bin.evaluate.text_to_speech -f experiments/your/checkpoint.pt \
                                               -s experiments/your/checkpoint.pt
"""
from pathlib import Path

import argparse
import logging
import random
import sys

from torchnlp.utils import pad_batch

import librosa
import numpy
import torch

from src import datasets
from src.hparams import configurable
from src.hparams import log_config
from src.hparams import set_hparams
from src.utils import Checkpoint
from src.utils import chunks
from src.utils import combine_signal
from src.utils import duplicate_stream

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@configurable
def main(spectrogram_model_checkpoint_path,
         signal_model_checkpoint_path,
         dataset=datasets.lj_speech_dataset,
         destination='results/',
         destination_stdout='stdout.log',
         destination_stderr='stderr.log',
         samples=25,
         device=torch.device('cpu'),
         batch_size=8,
         max_recursion=1000):
    """ Generate random samples of text to speech model to evaluate.

    Args:
        spectrogram_model_checkpoint_path (str): Spectrogram model checkpoint to load.
        signal_model_checkpoint_path (str): Signal model checkpoint to load.
        destination (str): Path to store results.
        destination_stdout (str): Filename of the stdout log.
        destination_stderr (str): Filename of the stderr log.
        sample_rate (int): Sample rate of audio evaluated.
        samples (int): Number of rows to evaluate.
        device (torch.device): Device on which to evaluate on.
        batch_size (int)
        max_recursion (int): The maximum sequential predictions to make with the spectrogram model.
    """
    destination = Path(destination)
    destination.mkdir(exist_ok=False)
    duplicate_stream(sys.stdout, destination / destination_stdout)
    duplicate_stream(sys.stderr, destination / destination_stderr)

    log_config()

    spectrogram_model_checkpoint_path = Path(spectrogram_model_checkpoint_path)
    signal_model_checkpoint_path = Path(signal_model_checkpoint_path)
    assert spectrogram_model_checkpoint_path.is_file()
    assert signal_model_checkpoint_path.is_file()

    spectrogram_model_checkpoint = Checkpoint.from_path(
        spectrogram_model_checkpoint_path, device=device)
    signal_model_checkpoint = Checkpoint.from_path(signal_model_checkpoint_path, device=device)
    logger.info('Loaded checkpoint: %s', spectrogram_model_checkpoint)
    logger.info('Loaded checkpoint: %s', signal_model_checkpoint)

    torch.set_grad_enabled(False)

    text_encoder = spectrogram_model_checkpoint.text_encoder
    spectrogram_model = spectrogram_model_checkpoint.model.eval().to(device)
    signal_model = signal_model_checkpoint.model.eval().to(device)

    train, dev = dataset()

    logger.info('Device: %s', device)
    logger.info('Number of Dev Rows: %d', len(dev))
    logger.info('Vocab Size: %d', text_encoder.vocab_size)
    logger.info('Spectrogram Model:\n%s' % spectrogram_model)
    logger.info('Signal Model:\n%s' % signal_model)

    sample = random.sample(list(range(len(dev))), samples)
    for chunk in chunks(sample, batch_size):
        batch = [dev[i] for i in chunk]
        logger.info('Batch size %d', len(batch))
        text_batch, text_length_batch = pad_batch(
            [text_encoder.encode(row['text']) for row in batch])
        text_batch = text_batch.transpose(0, 1).contiguous()

        logger.info('Running the spectrogram model...')
        _, predicted_spectrogram, _, _, lengths = spectrogram_model.infer(
            text_batch, max_recursion=max_recursion)

        logger.info('Running the signal model...')
        predicted_spectrogram = predicted_spectrogram.transpose(0, 1)
        predicted_coarse, predicted_fine, _ = signal_model.infer(predicted_spectrogram, pad=True)
        waveform_spectrogram_ratio = predicted_coarse.shape[1] / predicted_spectrogram.shape[1]
        assert (
            int(waveform_spectrogram_ratio) == waveform_spectrogram_ratio
        ), 'Waveform spectrogram ratio invariant %f' % waveform_spectrogram_ratio  # noqa E124

        for i in range(len(chunk)):
            waveform = combine_signal(predicted_coarse[i], predicted_fine[i])
            waveform = waveform[:int(lengths[i] * waveform_spectrogram_ratio)]

            gold_path = str(destination / ('%d_gold.wav' % chunk[i]))
            librosa.output.write_wav(gold_path, numpy.load(batch[i]['audio_path']))
            logger.info('Saved file %s', gold_path)

            predicted_path = str(destination / ('%d_predicted.wav' % chunk[i]))
            librosa.output.write_wav(predicted_path, waveform.numpy())
            logger.info('Saved file %s', predicted_path)
            print('-' * 100)


if __name__ == '__main__':  # pragma: no cover
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--signal_model', type=str, required=True, help='Signal model checkpoint to evaluate.')
    parser.add_argument(
        '--spectrogram_model',
        type=str,
        required=True,
        help='Spectrogram model checkpoint to evaluate.')
    cli_args = parser.parse_args()
    set_hparams()
    main(
        spectrogram_model_checkpoint_path=cli_args.spectrogram_model,
        signal_model_checkpoint_path=cli_args.signal_model)

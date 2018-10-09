"""
Generate random samples of text to speech model to evaluate.

Example:

    python3 -m src.bin.evaluate_text_to_speech -f experiments/your/checkpoint.pt \
                                               -s experiments/your/checkpoint.pt
"""
from pathlib import Path

import argparse
import logging
import random

from torchnlp.utils import pad_batch

import librosa
import torch

from src.bin.feature_model._utils import load_data
from src.bin.signal_model._utils import set_hparams
from src.utils import combine_signal
from src.utils import load_checkpoint

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def main(feature_model_checkpoint_path,
         signal_model_checkpoint_path,
         results_path='results/',
         samples=25,
         device=torch.device('cpu'),
         batch_size=2,
         max_recursion=1000):  # pragma: no cover
    """ Generate random samples of text to speech model to evaluate.

    Args:
        feature_model_checkpoint_path (str): Feature model checkpoint to load.
        signal_model_checkpoint_path (str): Signal model checkpoint to load.
        results_path (str): Path to store results.
        sample_rate (int): Sample rate of audio evaluated.
        samples (int): Number of rows to evaluate.
        device (torch.device): Device on which to evaluate on.
        batch_size (int)
        max_recursion (int): The maximum sequential predictions to make with the feature model.
    """
    results_path = Path(results_path)

    feature_model_checkpoint_path = Path(feature_model_checkpoint_path)
    assert feature_model_checkpoint_path.is_file()

    signal_model_checkpoint_path = Path(signal_model_checkpoint_path)
    assert signal_model_checkpoint_path.is_file()

    feature_model_checkpoint = load_checkpoint(feature_model_checkpoint_path, device=device)
    logger.info('Loaded checkpoint: %s', feature_model_checkpoint)

    signal_model_checkpoint = load_checkpoint(signal_model_checkpoint_path, device=device)
    logger.info('Loaded checkpoint: %s', signal_model_checkpoint)

    torch.set_grad_enabled(False)

    text_encoder = feature_model_checkpoint['text_encoder']
    feature_model = feature_model_checkpoint['model'].eval().to(device)
    signal_model = signal_model_checkpoint['model'].eval().to(device)

    _, dev, text_encoder = load_data(text_encoder=text_encoder, load_signal=True)

    logger.info('Device: %s', device)
    logger.info('Number of Dev Rows: %d', len(dev))
    logger.info('Vocab Size: %d', text_encoder.vocab_size)
    logger.info('Feature Model:\n%s' % feature_model)
    logger.info('Signal Model:\n%s' % signal_model)

    results_path.mkdir(exist_ok=False, parents=True)
    sample = random.sample(list(range(len(dev))), samples)
    for chunk in chunks(sample, batch_size):
        batch = [dev[i] for i in chunk]
        logger.info('Batch size %d', len(batch))
        text_batch, text_length_batch = pad_batch([row['text'] for row in batch])
        text_batch = text_batch.transpose(0, 1).contiguous()

        logger.info('Running the feature model...')
        _, predicted_spectrogram, _, _, lengths = feature_model.infer(
            text_batch, max_recursion=max_recursion)

        logger.info('Running the signal model...')
        predicted_spectrogram = predicted_spectrogram.transpose(0, 1)
        predicted_coarse, predicted_fine, _ = signal_model.infer(predicted_spectrogram, pad=True)
        waveform_spectrogram_ratio = predicted_coarse.shape[1] / predicted_spectrogram.shape[0]
        assert (int(waveform_spectrogram_ratio) == waveform_spectrogram_ratio
               ), 'Waveform spectrogram ratio invariant'  # noqa E124

        for i in range(len(chunk)):
            waveform = combine_signal(predicted_coarse[i], predicted_fine[i])
            waveform = waveform[:int(lengths[i] * waveform_spectrogram_ratio)]

            gold_path = str(results_path / ('%d_gold.wav' % chunk[i]))
            librosa.output.write_wav(gold_path, batch[i]['signal'].numpy())
            logger.info('Saved file %s', gold_path)

            predicted_path = str(results_path / ('%d_predicted.wav' % chunk[i]))
            librosa.output.write_wav(predicted_path, waveform.numpy())
            logger.info('Saved file %s', predicted_path)
            print('-' * 100)


if __name__ == '__main__':  # pragma: no cover
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-s',
        '--signal_model',
        type=str,
        required=True,
        help='Signal model checkpoint to evaluate.')
    parser.add_argument(
        '-f',
        '--feature_model',
        type=str,
        required=True,
        help='Feature model checkpoint to evaluate.')
    cli_args = parser.parse_args()
    set_hparams()
    main(
        feature_model_checkpoint_path=cli_args.feature_model,
        signal_model_checkpoint_path=cli_args.signal_model)

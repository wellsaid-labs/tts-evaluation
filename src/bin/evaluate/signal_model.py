"""
Generate random samples of signal model to evaluate.

Example:

    python3 -m src.bin.evaluate.signal_model --checkpoint experiments/your/checkpoint.pt
"""
from pathlib import Path

import argparse
import logging

import librosa
import torch

from src.bin.train.signal_model._data_iterator import RandomSampler
from src.bin.train.signal_model._utils import load_data
from src.bin.train.signal_model._utils import set_hparams
from src.utils import combine_signal
from src.utils import load_checkpoint
from src.utils.configurable import configurable

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@configurable
def main(checkpoint_path, results_path='results/', samples=25,
         device=torch.device('cpu')):  # pragma: no cover
    """ Generate random samples of signal model to evaluate.

    Args:
        checkpoint_path (str): Checkpoint to load.
        results_path (str): Path to store results.
        samples (int): Number of rows to evaluate.
        device (torch.device): Device on which to evaluate on.
    """
    results_path = Path(results_path)
    results_path.mkdir(exist_ok=False, parents=True)

    checkpoint_path = Path(checkpoint_path)
    assert checkpoint_path.is_file()

    checkpoint = load_checkpoint(checkpoint_path, device=device)
    logger.info('Loaded checkpoint: %s', checkpoint)

    train, dev = load_data()

    torch.set_grad_enabled(False)
    model = checkpoint['model'].eval().to(device)

    # TODO: Sample a batch and run this all at once.
    for i, j in enumerate(RandomSampler(dev)):
        if i >= samples:
            break

        logger.info('Evaluating dev row %d [%d of %d]', j, i + 1, samples)
        row = dev[j]

        # [batch_size, local_length, local_features_size]
        log_mel_spectrogram = row['log_mel_spectrogram'].unsqueeze(0).to(device)

        # [signal_length]
        signal = row['signal'].numpy()
        gold_path = str(results_path / ('%d_gold.wav' % j))
        librosa.output.write_wav(gold_path, signal)
        logger.info('Saved file %s', gold_path)

        predicted_coarse, predicted_fine, _ = model.infer(log_mel_spectrogram)
        predicted_signal = combine_signal(predicted_coarse.squeeze(0),
                                          predicted_fine.squeeze(0)).numpy()

        predicted_path = str(results_path / ('%d_predicted.wav' % j))
        librosa.output.write_wav(predicted_path, predicted_signal)
        logger.info('Saved file %s', predicted_path)
        print('-' * 100)


if __name__ == '__main__':  # pragma: no cover
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c', '--checkpoint', type=str, required=True, help='Signal model checkpoint to evaluate.')
    cli_args = parser.parse_args()

    set_hparams()
    main(checkpoint_path=cli_args.checkpoint)

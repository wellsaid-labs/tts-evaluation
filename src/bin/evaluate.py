"""
Generate random samples of signal model to evaluate from either predicted or

Example:

    python3 -m src.bin.evaluate --signal_model experiments/your/checkpoint.pt
"""
from pathlib import Path

import argparse
import logging

import librosa
import torch

from src import datasets
from src.audio import combine_signal
from src.hparams import configurable
from src.hparams import log_config
from src.hparams import set_hparams
from src.utils import Checkpoint
from src.utils import collate_sequences
from src.utils import evaluate
from src.utils import RandomSampler
from src.utils import record_stream
from src.utils import tensors_to
from src.datasets import compute_spectrograms

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@configurable
def main(signal_model_checkpoint_path,
         spectrogram_model_checkpoint_path=None,
         dataset=datasets.lj_speech_dataset,
         destination='results/',
         num_samples=25):
    """ Generate random samples of signal model to evaluate.

    Args:
        signal_model_checkpoint_path (str): Checkpoint used to predict a raw waveform given a
            spectrogram.
        spectrogram_model_checkpoint_path (str, optional): Checkpoint used to generate spectrogram
            from text as input to the signal model.
        dataset (callable, optional): Callable that returns an iterable of ``dict``.
        destination (str): Path to store results.
        num_samples (int): Number of rows to evaluate.
    """
    destination = Path(destination)
    destination.mkdir(exist_ok=False, parents=True)
    record_stream(destination)

    log_config()

    signal_model_checkpoint = Checkpoint.from_path(
        Path(signal_model_checkpoint_path), device=torch.device('cpu'))
    _, dev = dataset()

    # Sample and batch the validation data
    use_predicted_spectrogram = spectrogram_model_checkpoint_path is not None
    indicies = list(RandomSampler(dev))[:num_samples]
    examples = compute_spectrograms(
        [dev[i] for i in indicies],
        checkpoint_path=spectrogram_model_checkpoint_path,
        device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'),
        on_disk=False)
    batch = collate_sequences(examples, padding_index=0)
    batch = tensors_to(batch, device=torch.device('cpu'), non_blocking=True)
    spectrogram = batch.predicted_spectrogram if use_predicted_spectrogram else batch.spectrogram

    with evaluate(signal_model_checkpoint.model):
        # [batch_size, local_length, local_features_size] → [batch_size, signal_length]
        predicted_coarse, predicted_fine, _ = signal_model_checkpoint.model.infer(spectrogram[0])
        predicted_signal = combine_signal(predicted_coarse, predicted_fine).numpy()

    for i, example, predicted_signal in zip(indicies, examples, predicted_signal.split(1)):
        gold_path = str(destination / ('%d_gold.wav' % i))
        librosa.output.write_wav(gold_path, example.spectrogram_audio)
        logger.info('Saved file %s', gold_path)

        predicted_path = str(destination / ('%d_predicted.wav' % i))
        librosa.output.write_wav(predicted_path, predicted_signal)
        logger.info('Saved file %s', predicted_path)
        logger.info('-' * 100)


if __name__ == '__main__':  # pragma: no cover
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--signal_model', type=str, required=True, help='Signal model checkpoint to evaluate.')
    parser.add_argument(
        '--spectrogram_model',
        type=str,
        default=None,
        help='Spectrogram model checkpoint to evaluate.')
    cli_args = parser.parse_args()
    set_hparams()
    main(
        signal_model_checkpoint_path=cli_args.signal_model,
        spectrogram_model_checkpoint_path=cli_args.spectrogram_model)

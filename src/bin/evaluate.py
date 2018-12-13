"""
Generate random samples of signal model to evaluate from either predicted or

Example:

    python3 -m src.bin.evaluate --signal_model experiments/your/checkpoint.pt
"""
from pathlib import Path

import argparse
import logging

# NOTE: Needs to be imported before torch
# Remove after this issue is resolved https://github.com/comet-ml/issue-tracking/issues/178
import comet_ml  # noqa

import librosa
import torch
import numpy

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
from src.utils import chunks

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@configurable
def main(signal_model_checkpoint_path,
         spectrogram_model_checkpoint_path=None,
         dataset=datasets.lj_speech_dataset,
         destination='results/',
         num_samples=25,
         aligned=False,
         signal_model_batch_size=4):
    """ Generate random samples of signal model to evaluate.

    Args:
        signal_model_checkpoint_path (str): Checkpoint used to predict a raw waveform given a
            spectrogram.
        spectrogram_model_checkpoint_path (str, optional): Checkpoint used to generate spectrogram
            from text as input to the signal model.
        dataset (callable, optional): Callable that returns an iterable of ``dict``.
        destination (str): Path to store results.
        num_samples (int): Number of rows to evaluate.
        aligned (bool): If ``True``, predict a ground truth aligned spectrogram.
        signal_model_batch_size (int): The batch size for the signal model. This is lower
            than during training because we are no longer using small slices.
    """
    destination = Path(destination)
    destination.mkdir(exist_ok=False, parents=True)
    record_stream(destination)

    log_config()

    # NOTE: On CUDA, we run out of memory quickly and it's slow to iterate. While on CPU, this
    # is not a problem.
    signal_model_checkpoint = Checkpoint.from_path(
        Path(signal_model_checkpoint_path), device=torch.device('cpu'))
    _, dev = dataset()

    # Sample and batch the validation data
    use_predicted = spectrogram_model_checkpoint_path is not None
    indicies = list(RandomSampler(dev))[:num_samples]
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    examples = compute_spectrograms([dev[i] for i in indicies],
                                    checkpoint_path=spectrogram_model_checkpoint_path,
                                    device=device,
                                    on_disk=False,
                                    aligned=aligned)

    def get_spectrogram_length(example):
        return (example.predicted_spectrogram.shape[0]
                if use_predicted else example.spectrogram.shape[0])

    examples = sorted(examples, key=lambda e: -get_spectrogram_length(e))
    for examples_chunk in chunks(examples, signal_model_batch_size):
        batch = collate_sequences(examples_chunk, padding_index=0)
        batch = tensors_to(batch, device=torch.device('cpu'), non_blocking=True)
        spectrogram = (batch.predicted_spectrogram if use_predicted else batch.spectrogram)

        logger.info('Predicting signal...')
        with evaluate(signal_model_checkpoint.model):
            # [batch_size, local_length, local_features_size] → [batch_size, signal_length]
            predicted_coarse, predicted_fine, _ = signal_model_checkpoint.model.infer(
                spectrogram[0], use_tqdm=True)
            predicted_signal = combine_signal(predicted_coarse, predicted_fine).numpy()

        # Split and save
        factor = int(predicted_signal.shape[1] / spectrogram[0].shape[1])
        splits = numpy.split(predicted_signal, signal_model_batch_size)
        for i, example, predicted, spectrogram_length in zip(indicies, examples_chunk, splits,
                                                             spectrogram[1]):
            gold_path = str(destination / ('%d_gold.wav' % i))
            librosa.output.write_wav(gold_path, example.spectrogram_audio.numpy())
            logger.info('Saved file %s', gold_path)

            predicted_path = str(destination / ('%d_predicted.wav' % i))
            librosa.output.write_wav(predicted_path, predicted[0, :spectrogram_length * factor])
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

"""
Generate random samples of signal model to evaluate from either predicted or

Example:

    python3 -m src.bin.evaluate --signal_model experiments/your/checkpoint.pt
"""
from functools import partial
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
from src.audio import griffin_lim
from src.datasets import compute_spectrograms
from src.datasets import Speaker
from src.hparams import configurable
from src.hparams import log_config
from src.hparams import set_hparams
from src.utils import Checkpoint
from src.utils import chunks
from src.utils import collate_sequences
from src.utils import evaluate
from src.utils import RandomSampler
from src.utils import record_stream
from src.utils import tensors_to

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _save(destination, index, example, predicted_waveform):
    """ Save a gold and predicted example.

    Args:
        destination (Path): Destination to save the predicted waveform.
        index (int): Row index used to save the filename.
        example (SpectrogramTextSpeechRow): The initial spectrogram example used to predict
            the waveform.
        predicted_waveform (np.ndarray): 1D signal.
    """
    speaker_name = example.speaker.name.lower().replace(' ', '_')
    gold_path = str(destination / ('%d_%s_gold.wav' % (index, speaker_name)))
    librosa.output.write_wav(gold_path, example.spectrogram_audio.numpy())
    logger.info('Saved file %s', gold_path)

    predicted_path = str(destination / ('%d_%s_predicted.wav' % (index, speaker_name)))
    librosa.output.write_wav(predicted_path, predicted_waveform)
    logger.info('Saved file %s', predicted_path)


def _get_spectrogram_length(example, use_predicted):
    return (example.predicted_spectrogram.shape[0]
            if use_predicted else example.spectrogram.shape[0])


@configurable
def main(signal_model_checkpoint_path=None,
         spectrogram_model_checkpoint_path=None,
         dataset=datasets.lj_speech_dataset,
         destination='results/',
         num_samples=32,
         aligned=False,
         speaker=Speaker.HILARY_NORIEGA,
         spectrogram_model_batch_size=1,
         signal_model_batch_size=1):
    """ Generate random samples of signal model to evaluate.

    NOTE: We use a batch size of 1 during evaluation to get similar results as the deployed product.
    Padding needed to batch together sequences, affects both the signal model and the spectrogram
    model.

    Args:
        signal_model_checkpoint_path (str, optional): Checkpoint used to predict a raw waveform
            given a spectrogram.
        spectrogram_model_checkpoint_path (str, optional): Checkpoint used to generate spectrogram
            from text as input to the signal model.
        dataset (callable, optional): Callable that returns an iterable of ``dict``.
        destination (str, optional): Path to store results.
        num_samples (int, optional): Number of rows to evaluate.
        aligned (bool, optional): If ``True``, predict a ground truth aligned spectrogram.
        speaker (Speaker, optional): Filter the data for a particular speaker.
        spectrogram_model_batch_size (int, optional)
        signal_model_batch_size (int, optional): The batch size for the signal model. This is lower
            than during training because we are no longer using small slices.
        signal_model_device (torch.device, optional): Device used for signal model inference, note
            that on CPU the signal model tends to run faster and does not run out of memory.
    """
    destination = Path(destination)
    destination.mkdir(exist_ok=False, parents=True)
    record_stream(destination)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    log_config()

    # Sample and batch the validation data
    _, dev = dataset()

    if speaker is not None:
        dev = [r for r in dev if r.speaker == speaker]

    indicies = list(RandomSampler(dev))
    if num_samples is not None:
        indicies = indicies[:num_samples]

    examples = compute_spectrograms([dev[i] for i in indicies],
                                    checkpoint_path=spectrogram_model_checkpoint_path,
                                    device=device,
                                    on_disk=False,
                                    aligned=aligned,
                                    batch_size=spectrogram_model_batch_size)

    if signal_model_checkpoint_path is None:
        for i, example in zip(indicies, examples):
            waveform = griffin_lim(example.predicted_spectrogram.numpy())
            _save(destination, i, example, waveform)
    else:
        signal_model_checkpoint = Checkpoint.from_path(
            Path(signal_model_checkpoint_path), device=device)
        signal_model_inferrer = signal_model_checkpoint.model.to_inferrer(device=device)
        use_predicted = spectrogram_model_checkpoint_path is not None

        # NOTE: Sort by spectrogram lengths to batch similar sized outputs together
        _get_length_partial = partial(_get_spectrogram_length, use_predicted=use_predicted)
        examples = sorted(examples, key=lambda e: -_get_length_partial(e))

        for chunk in chunks(list(zip(examples, indicies)), signal_model_batch_size):
            examples_chunk, indicies_chunk = zip(*chunk)
            batch = collate_sequences(examples_chunk, padding_index=0)
            batch = tensors_to(batch, device=device, non_blocking=True)
            spectrogram = (batch.predicted_spectrogram if use_predicted else batch.spectrogram)

            logger.info('Predicting signal...')
            with evaluate(signal_model_inferrer):
                # [batch_size, local_length, local_features_size] → [batch_size, signal_length]
                predicted_coarse, predicted_fine, _ = signal_model_inferrer(
                    spectrogram[0], use_tqdm=True)
                predicted_signal = combine_signal(predicted_coarse, predicted_fine).numpy()

            # Split and save
            factor = int(predicted_signal.shape[1] / spectrogram[0].shape[1])
            splits = numpy.split(predicted_signal, signal_model_batch_size)
            for i, example, predicted, spectrogram_length in zip(indicies_chunk, examples_chunk,
                                                                 splits, spectrogram[1]):
                _save(destination, i, example, predicted[0, :spectrogram_length * factor])
                logger.info('-' * 100)


if __name__ == '__main__':  # pragma: no cover
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--signal_model', type=str, default=None, help='Signal model checkpoint to evaluate.')
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

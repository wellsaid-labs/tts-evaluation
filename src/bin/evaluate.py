"""
Generate random samples of signal model to evaluate from either predicted or

Example:

    python3 -m src.bin.evaluate --signal_model experiments/your/checkpoint.pt
"""
from functools import partial
from pathlib import Path

import argparse
import logging

from torchnlp.utils import tensors_to
from torch.utils.data.sampler import RandomSampler

import torch
import scipy

from src.audio import combine_signal
from src.audio import griffin_lim
from src.datasets import balance_dataset
from src.datasets import compute_spectrograms
from src.datasets import TextSpeechRow
from src.hparams import configurable
from src.hparams import ConfiguredArg
from src.hparams import log_config
from src.hparams import set_hparams
from src.utils import Checkpoint
from src.utils import evaluate
from src.utils import record_stream

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _save(destination, tags, speaker, waveform):
    """ Save a waveform.

    Args:
        destination (Path): Destination to save the predicted waveform.
        tags (list of str): Tags to add to the filename.
        speaker (src.datasets.Speaker): Speaker of the waveform.
        waveform (np.ndarray): 1D signal to save.
    """
    speaker_name = speaker.name.lower().replace(' ', '_')
    path = str(destination / ('%s,%s.wav' % (speaker_name, ','.join(tags))))
    scipy.io.wavfile.write(filename=path, data=waveform)
    logger.info('Saved file "%s" with waveform of shape `%s` and dtype `%s`', path, waveform.shape,
                waveform.dtype)


def _get_spectrogram_length(example, use_predicted):
    return (example.predicted_spectrogram.shape[0]
            if use_predicted else example.spectrogram.shape[0])


def _sample_dataset(dataset, speaker_encoder, num_samples=None, speaker=None, balanced=False):
    """ Given a dataset, sample from it and configure it.

    Args:
        dataset (callable or list): Dataset returned from `src.datasets` or a `list` of `str`.
            Where the `dev` set is the second value returned.
        speaker_encoder (torchnlp.TextEncoder): Speaker encoder with all possible speakers.
        num_samples (int or None, optional): Randomly sample a number of samples.
        speaker (src.datasets.Speaker): Filter to only one speaker.
        balanced (bool, optional): If ``True``, sample from a dataset with equal speaker
            distribution.

    Returns:
        dataset (list of TextSpeechRow)
        indicies (list of int): Index of each example, useful to identifying the example.
    """
    if callable(dataset):  # CASE: Dataset created by a callable
        _, dev = dataset()

        if balanced:
            dev = balance_dataset(dev, lambda r: r.speaker)
        if speaker is not None:
            dev = [r for r in dev if r.speaker == speaker]

        indicies = list(RandomSampler(dev))
        if num_samples is not None:
            indicies = indicies[:num_samples]

        ret = [dev[i] for i in indicies]
    elif isinstance(dataset, list):  # CASE: List of text
        speakers = speaker_encoder.vocab if speaker is None else [speaker]
        ret = []
        for speaker in speakers:
            for text in dataset:
                ret.append(
                    TextSpeechRow(text=text, speaker=speaker, audio_path=None, metadata=None))
        indicies = list(range(len(ret)))
    else:
        raise TypeError()

    return ret, indicies


@configurable
def main(dataset=ConfiguredArg(),
         signal_model_checkpoint_path=None,
         spectrogram_model_checkpoint_path=None,
         destination='results/',
         num_samples=50,
         aligned=False,
         speaker=None,
         balanced=False,
         spectrogram_model_batch_size=1,
         signal_model_device=torch.device('cpu')):
    """ Generate random samples of signal model to evaluate.

    NOTE: We use a batch size of 1 during evaluation to get similar results as the deployed product.
    Padding needed to batch together sequences, affects both the signal model and the spectrogram
    model.

    Args:
        dataset (callable): Callable that returns an iterable of ``dict``.
        signal_model_checkpoint_path (str, optional): Checkpoint used to predict a raw waveform
            given a spectrogram.
        spectrogram_model_checkpoint_path (str, optional): Checkpoint used to generate spectrogram
            from text as input to the signal model.
        destination (str, optional): Path to store results.
        num_samples (int, optional): Number of rows to evaluate.
        aligned (bool, optional): If ``True``, predict a ground truth aligned spectrogram.
        speaker (Speaker, optional): Filter the data for a particular speaker.
        balanced (bool, optional): If ``True``, sample from a dataset with equal speaker
            distribution. Note, that this operation shuffles the rows.
        spectrogram_model_batch_size (int, optional)
        signal_model_device (torch.device, optional): Device used for signal model inference, note
            that on CPU the signal model does not run out of memory.
    """
    # Record the standard streams
    destination = Path(destination)
    destination.mkdir(exist_ok=False, parents=True)
    record_stream(destination)

    log_config()

    # Create the dataset to iterate over
    spectrogram_model_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    spectrogram_model_checkpoint = Checkpoint.from_path(
        spectrogram_model_checkpoint_path, device=spectrogram_model_device)
    examples, indicies = _sample_dataset(
        dataset,
        speaker_encoder=spectrogram_model_checkpoint.input_encoder.speaker_encoder,
        num_samples=num_samples,
        speaker=speaker,
        balanced=balanced)

    # Compute target and / or predict spectrograms
    examples = compute_spectrograms(
        examples,
        checkpoint_path=spectrogram_model_checkpoint_path,
        device=spectrogram_model_device,
        on_disk=False,
        aligned=aligned,
        batch_size=spectrogram_model_batch_size)

    # Output griffin-lim
    for i, example in zip(indicies, examples):
        waveform = griffin_lim(example.predicted_spectrogram.numpy())
        _save(destination, ['index_%d' % i, 'griffin_lim'], example.speaker, waveform)
        if example.spectrogram_audio is not None:
            _save(destination, ['index_%d' % i, 'target'], example.speaker,
                  example.spectrogram_audio.numpy())

    if signal_model_checkpoint_path is None:
        return

    # Predict with the signal model
    signal_model_checkpoint = Checkpoint.from_path(
        signal_model_checkpoint_path, device=signal_model_device)
    signal_model_inferrer = signal_model_checkpoint.model.to_inferrer(device=signal_model_device)
    use_predicted = spectrogram_model_checkpoint_path is not None

    # NOTE: Sort by spectrogram lengths to batch similar sized outputs together
    _get_length_partial = partial(_get_spectrogram_length, use_predicted=use_predicted)
    examples = list(zip(examples, indicies))
    examples = sorted(examples, key=lambda e: _get_length_partial(e[0]), reverse=True)

    for example, index in examples:
        example = tensors_to(example, device=signal_model_device, non_blocking=True)
        spectrogram = (example.predicted_spectrogram if use_predicted else example.spectrogram)
        logger.info('Predicting signal a %s spectrogram.', spectrogram.shape)
        with evaluate(signal_model_inferrer):
            # [local_length, local_features_size] → [signal_length]
            predicted_coarse, predicted_fine, _ = signal_model_inferrer(spectrogram)
            predicted_signal = combine_signal(predicted_coarse, predicted_fine, return_int=True)
            predicted_signal = predicted_signal.numpy()

        # Save
        _save(destination, ['index_' + index, 'wave_rnn'], example.speaker, predicted_signal)
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
    parser.add_argument(
        '-t',
        '--text',
        action='append',
        help='Input custom text to the model to compute.',
        default=None)
    args = parser.parse_args()
    kwargs = {
        'signal_model_checkpoint_path': args.signal_model,
        'spectrogram_model_checkpoint_path': args.spectrogram_model
    }
    if args.text is not None:
        kwargs['dataset'] = args.text
    set_hparams()
    main(**kwargs)

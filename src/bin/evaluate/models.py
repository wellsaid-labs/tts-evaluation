"""
Generate random samples of `dev` dataset to evaluate.

TODO: Rename to `evaluate_batch.py` in contrast to `evaluate_stream.py`.

Examples
--------

Example of sampling the preprocessed dataset:
    $ python -m src.bin.evaluate.models


Example of generating signal model samples from the ground truth spectrograms:
    $ python -m src.bin.evaluate.models --signal_model experiments/your/checkpoint.pt


Example of generating TTS samples end-to-end:
    $ python -m src.bin.evaluate.models --signal_model experiments/your/checkpoint.pt \
                                        --spectrogram_model experiments/your/checkpoint.pt


Example of generating TTS samples end-to-end with custom text:
    $ python -m src.bin.evaluate.models --signal_model experiments/your/checkpoint.pt \
                                        --spectrogram_model experiments/your/checkpoint.pt \
                                        --text "custom text" \
                                        --text "more custom text"
"""
from itertools import product
from pathlib import Path

import argparse
import logging
import os
import time

from hparams import configurable
from hparams import HParam
from hparams import log_config
from torchnlp.random import fork_rng
from torchnlp.samplers import BalancedSampler
from torchnlp.utils import tensors_to

import pandas
import torch

# NOTE: Some modules log on import; therefore, we first setup logging.
from src.environment import set_basic_logging_config

set_basic_logging_config()

from src.audio import griffin_lim
from src.audio import write_audio
from src.datasets import add_predicted_spectrogram_column
from src.datasets import add_spectrogram_column
from src.datasets import TextSpeechRow
from src.environment import SAMPLES_PATH
from src.hparams import set_hparams
from src.utils import bash_time_label
from src.utils import Checkpoint
from src.utils import RecordStandardStreams

logger = logging.getLogger(__name__)


def _save(destination, tags, speaker, waveform, obscure=False):
    """ Save a waveform.

    Args:
        destination (Path): Destination to save the predicted waveform.
        tags (list of str): Tags to add to the filename.
        speaker (src.datasets.Speaker): Speaker of the waveform.
        waveform (np.ndarray): 1D signal to save.
        obscure (bool, optional): Hash the filename, obscuring the orignal filename.

    Returns:
        (str): The filename of the saved file.
    """
    speaker_name = speaker.name.lower().replace(' ', '_')
    filename = 'speaker=%s,%s' % (speaker_name, ','.join(tags))
    # NOTE: The Python `hash` built-in is conditioned on the process; therefore, it generates
    # unique hashes per process.
    filename = '%x' % hash(filename) if obscure else filename
    filename_with_suffix = str(filename) + '.wav'
    collision = 1
    while (destination / filename_with_suffix).exists():
        filename_with_suffix = str(filename) + ',collision=%d.wav' % collision
        collision += 1
    path = str(destination / filename_with_suffix)
    write_audio(path, waveform)
    logger.info('Saved file "%s" with waveform of shape `%s` and dtype `%s`', path, waveform.shape,
                waveform.dtype)
    return filename_with_suffix


@configurable
def _get_dev_dataset(dataset=HParam()):
    _, dev = dataset()
    return dev


@configurable
def _get_sample_rate(sample_rate=HParam()):
    return sample_rate


def main(dataset,
         signal_model_checkpoint,
         spectrogram_model_checkpoint,
         num_samples,
         name='',
         get_sample_rate=_get_sample_rate,
         destination=SAMPLES_PATH / bash_time_label(),
         metadata_filename='metadata.csv',
         aligned=False,
         speakers=None,
         balanced=True,
         obscure=False,
         no_target_audio=False,
         no_griffin_lim=False,
         no_signal_model=False,
         spectrogram_model_batch_size=1,
         spectrogram_model_device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
         random_seed=os.getpid()):
    """ Generate random samples of the `dataset`.

    Args:
        dataset (callable): Callable that returns an iterable of `dict`.
        signal_model_checkpoint (str or None): Checkpoint used to predict a raw waveform
            given a spectrogram.
        spectrogram_model_checkpoint (str or None): Checkpoint used to generate spectrogram
            from text as input to the signal model.
        num_samples (int): Number of rows to evaluate.
        name (str, optional): The name of this evaluation process to be included in the metadata.
        get_sample_rate (callable, optional): Get the number of samples in a clip per second.
        destination (str, optional): Path to store results.
        metadata_filename (str, optional): The filename for a CSV file containing clip metadata.
        aligned (bool, optional): If `True`, predict a ground truth aligned spectrogram.
        speakers (list of Speaker, optional): Filter the data for a particular speakers.
        balanced (bool, optional): If `True`, sample from a dataset with equal speaker
            distribution. Note, that this operation shuffles the rows.
        obscure (bool, optional): If `True`, obscure the audio filename such that the
            filename does not provide hints towards the method of synthesis.
        no_target_audio (bool, optional): If `True` don't save target audio clips.
        no_griffin_lim (bool, optional): If `True` don't generate and save griffin lim clips.
        no_signal_model (bool, optional): If `True` don't generate and save signal model clips.
        spectrogram_model_batch_size (int, optional)
        spectrogram_model_device (torch.device, optional): Device used for spectrogram model
            inference.
        random_seed (int, optional): Random seed determining which `dev` rows are randomly
            evaluated.
    """
    destination = Path(destination)
    destination.mkdir(exist_ok=False)

    RecordStandardStreams(destination).start()

    log_config()

    sample_rate = get_sample_rate()

    # Sample from the dataset
    dataset = dataset() if callable(dataset) else dataset
    dataset = list(
        dataset if speakers is None else filter(lambda r: r.speaker in speakers, dataset))
    logger.info('The random seed is: %s', random_seed)
    with fork_rng(seed=random_seed):
        indicies = (
            list(BalancedSampler(dataset, get_class=lambda r: r.speaker, num_samples=num_samples))
            if balanced else range(len(dataset)))
    dataset = [dataset[i] for i in indicies]

    has_target_audio = all(e.audio_path is not None for e in dataset)

    # Metadata saved along with audio clips
    metadata = []
    # NOTE: `os.getpid` is often used by routines that generate unique identifiers, learn more:
    # http://manpages.ubuntu.com/manpages/cosmic/man2/getpid.2.html
    add_to_metadata = lambda example, **kwargs: metadata.append(
        dict(
            **kwargs,
            text=example.text,
            name=name,
            speaker=example.speaker.name,
            signal_model_checkpoint_path=(None if signal_model_checkpoint is None else
                                          signal_model_checkpoint.path),
            spectrogram_model_checkpoint_path=(None if spectrogram_model_checkpoint is None else
                                               spectrogram_model_checkpoint.path),
            spectrogram_model_batch_size=spectrogram_model_batch_size,
            is_aligned=aligned,
            is_balanced=balanced,
            process_id=os.getpid()))
    _save_partial = lambda i, tags, *args: _save(
        destination, ['example_index=%d' % i] + tags, *args, obscure=obscure)

    # Save the target predictions
    if has_target_audio:
        # NOTE: Adds `spectrogram_audio` and `spectrogram` column.
        dataset = add_spectrogram_column(dataset, on_disk=False)
        if not no_target_audio:
            for i, example in zip(indicies, dataset):
                waveform = example.spectrogram_audio.cpu().numpy()
                audio_path = _save_partial(i, ['type=gold'], example.speaker, waveform)
                add_to_metadata(
                    example,
                    audio_length_in_seconds=waveform.shape[0] / sample_rate,
                    audio_path=audio_path,
                    example_index=i,
                    type='gold')
    else:
        logger.info('Skipping the writing of ground truth audio.')

    # Save the griffin-lim predictions
    if spectrogram_model_checkpoint is not None:
        logger.info('The spectrogram model path is: %s', spectrogram_model_checkpoint.path)
        dataset = add_predicted_spectrogram_column(
            dataset,
            spectrogram_model_checkpoint,
            spectrogram_model_device,
            batch_size=spectrogram_model_batch_size,
            aligned=aligned,
            on_disk=False)
        if not no_griffin_lim:
            # TODO: Consider predicting `griffin_lim` for real spectrogram.
            for i, example in zip(indicies, dataset):
                waveform = griffin_lim(example.predicted_spectrogram.cpu().numpy())
                audio_path = _save_partial(i, ['type=griffin_lim'], example.speaker, waveform)
                add_to_metadata(
                    example,
                    audio_length_in_seconds=waveform.shape[0] / sample_rate,
                    audio_path=audio_path,
                    example_index=i,
                    type='griffin_lim')
    else:
        logger.info('Skipping the writing of griffin-lim predictions.')

    # Save the signal model predictions
    if not no_signal_model and signal_model_checkpoint is not None and (
            has_target_audio or spectrogram_model_checkpoint is not None):
        logger.info('The signal model path is: %s', signal_model_checkpoint.path)
        logger.info('Running inference with %d threads.', torch.get_num_threads())
        # TODO: Factor out removing `exponential_moving_parameter_average` into the signal model's
        # inference mode.
        signal_model_checkpoint.exponential_moving_parameter_average.apply_shadow()
        signal_model = signal_model_checkpoint.model.eval()
        use_predicted = spectrogram_model_checkpoint is not None

        # NOTE: Sort by spectrogram lengths to batch similar sized outputs together
        iterator = list(zip(dataset, indicies))
        get_length = lambda e: getattr(e[0], 'predicted_spectrogram'
                                       if use_predicted else 'spectrogram').shape[0]
        iterator = sorted(iterator, key=get_length)

        for example, i in iterator:
            example = tensors_to(example, device=torch.device('cpu'), non_blocking=True)
            spectrogram = example.predicted_spectrogram if use_predicted else example.spectrogram
            logger.info('Predicting signal from spectrogram of size %s.', spectrogram.shape)
            start = time.time()
            # [local_length, local_features_size] → [signal_length]
            with torch.no_grad():
                waveform = signal_model(spectrogram)
            logger.info('Processed in %fx real time.',
                        (time.time() - start) / (waveform.shape[0] / sample_rate))

            audio_path = _save_partial(i, ['type=signal_model'], example.speaker, waveform)
            add_to_metadata(
                example,
                audio_length_in_seconds=waveform.shape[0] / sample_rate,
                audio_path=audio_path,
                example_index=i,
                type='signal_model')
            logger.info('-' * 100)
    else:
        logger.info('Skipping the writing of neural vocoder predictions.')

    pandas.DataFrame(metadata).to_csv(str((destination / metadata_filename)), index=False)


if __name__ == '__main__':  # pragma: no cover
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='', help='A name for this evaluation process.')
    parser.add_argument(
        '--signal_model', type=str, default=None, help='Signal model checkpoint to evaluate.')
    parser.add_argument(
        '--spectrogram_model',
        type=str,
        default=None,
        help='Spectrogram model checkpoint to evaluate.')
    parser.add_argument(
        '--text', action='append', help='Input custom text to the model to compute.', default=None)
    parser.add_argument(
        '--num_samples', type=int, help='Number of samples to generate in total.', default=50)
    parser.add_argument(
        '--no_target_audio',
        action='store_true',
        default=False,
        help='Do not save target audio clips.')
    parser.add_argument(
        '--no_griffin_lim',
        action='store_true',
        default=False,
        help='Do not generate and save griffin lim clips.')
    parser.add_argument(
        '--no_signal_model',
        action='store_true',
        default=False,
        help='Do not generate and save signal model clips.')
    parser.add_argument(
        '--obscure_filename',
        action='store_true',
        default=False,
        help='Obscure the filename such that the audio file\'s generation method is unknowable.')
    args = parser.parse_args()

    # NOTE: Load early and crash early by ensuring that the checkpoint exists and is not corrupt.
    if args.signal_model is not None:
        args.signal_model = Checkpoint.from_path(args.signal_model)

    dataset = _get_dev_dataset
    if args.spectrogram_model is not None:
        args.spectrogram_model = Checkpoint.from_path(args.spectrogram_model)
        if args.text is not None:
            speakers = args.spectrogram_model.input_encoder.speaker_encoder.vocab
            dataset = [TextSpeechRow(t, s, None) for t, s in product(args.text, speakers)]
    elif args.text is not None:
        raise argparse.ArgumentTypeError(
            'For custom data, `--spectrogram_model` must be defined; '
            'otherwise, there is no accompanying audio data for evaluation.')

    set_hparams()

    main(
        dataset=dataset,
        destination=SAMPLES_PATH / args.name if args.name else SAMPLES_PATH / bash_time_label(),
        name=args.name,
        spectrogram_model_checkpoint=args.spectrogram_model,
        signal_model_checkpoint=args.signal_model,
        balanced=callable(dataset),
        num_samples=args.num_samples if args.text is None else len(dataset),
        no_griffin_lim=args.no_griffin_lim,
        no_signal_model=args.no_signal_model,
        no_target_audio=args.no_target_audio,
        obscure=args.obscure_filename)

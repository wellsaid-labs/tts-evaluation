"""
Generate random samples of `dev` dataset to evaluate.

Examples
--------

Example of sampling the preprocessed dataset:
    $ python3 -m src.bin.evaluate


Example of generating signal model samples from the ground truth spectrograms:
    $ python3 -m src.bin.evaluate --signal_model experiments/your/checkpoint.pt


Example of generating TTS samples end-to-end:
    $ python3 -m src.bin.evaluate --signal_model experiments/your/checkpoint.pt \
    >                             --spectrogram_model experiments/your/checkpoint.pt


Example of generating TTS samples end-to-end with custom text:
    $ python3 -m src.bin.evaluate --signal_model experiments/your/checkpoint.pt \
    >                             --spectrogram_model experiments/your/checkpoint.pt \
    >                             --text "custom text" \
    >                             --text "more custom text"
"""
from itertools import product
from pathlib import Path

import argparse
import logging
import sys

from torch.utils.data.sampler import RandomSampler
from torchnlp.utils import tensors_to

import pandas
import torch

# NOTE: Some modules log on import; therefore, we first setup logging.
from src.environment import set_basic_logging_config

set_basic_logging_config()

from src.audio import combine_signal
from src.audio import griffin_lim
from src.audio import write_audio
from src.datasets import add_predicted_spectrogram_column
from src.datasets import add_spectrogram_column
from src.datasets import TextSpeechRow
from src.hparams import configurable
from src.hparams import ConfiguredArg
from src.hparams import log_config
from src.hparams import set_hparams
from src.utils import balance_list
from src.utils import Checkpoint
from src.utils import evaluate
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
    if obscure:
        filename = hash(filename)
    filename = str(filename) + '.wav'
    path = str(destination / filename)
    write_audio(path, waveform)
    logger.info('Saved file "%s" with waveform of shape `%s` and dtype `%s`', path, waveform.shape,
                waveform.dtype)
    return filename


@configurable
def _get_dev_dataset(dataset=ConfiguredArg()):
    _, dev = dataset()
    return dev


def main(dataset,
         signal_model_checkpoint,
         spectrogram_model_checkpoint,
         num_samples,
         destination='samples/',
         metadata_filename='metadata.csv',
         aligned=False,
         speakers=None,
         balanced=True,
         obscure=False,
         spectrogram_model_batch_size=1,
         spectrogram_model_device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    """ Generate random samples of the `dataset`.

    NOTE: Padding can affect the output of the signal and spectrogram model; therefore, it's
    recommended to use a `batch_size` of one to ensure similar results as the deployed product.

    Args:
        dataset (callable): Callable that returns an iterable of `dict`.
        signal_model_checkpoint (str or None): Checkpoint used to predict a raw waveform
            given a spectrogram.
        spectrogram_model_checkpoint (str or None): Checkpoint used to generate spectrogram
            from text as input to the signal model.
        num_samples (int): Number of rows to evaluate.
        destination (str, optional): Path to store results.
        metadata_filename (str, optional): The filename for a CSV file containing clip metadata.
        aligned (bool, optional): If `True`, predict a ground truth aligned spectrogram.
        speakers (list of Speaker, optional): Filter the data for a particular speakers.
        balanced (bool, optional): If `True`, sample from a dataset with equal speaker
            distribution. Note, that this operation shuffles the rows.
        obscure (bool, optional): If `True`, obscure the audio filename such that the
            filename does not provide hints towards the method of synthesis.
        spectrogram_model_batch_size (int, optional)
        spectrogram_model_device (torch.device, optional): Device used for spectrogram model
            inference.
    """
    destination = Path(destination)
    destination.mkdir(exist_ok=False, parents=True)

    RecordStandardStreams(destination).start()

    logger.info('The command line arguments are: %s', str(sys.argv))

    log_config()

    # Sample from the dataset
    dataset = dataset() if callable(dataset) else dataset
    dataset = balance_list(dataset, lambda r: r.speaker) if balanced else dataset
    dataset = dataset if speakers is None else filter(lambda r: r.speaker in speakers, dataset)
    indicies = list(RandomSampler(dataset))[:num_samples] if num_samples else range(len(dataset))
    dataset = [dataset[i] for i in indicies]

    has_target_audio = all(e.audio_path is not None for e in dataset)

    # Metadata saved along with audio clips
    metadata = []
    add_to_metadata = lambda example, **kwargs: metadata.append(
        dict(**kwargs, text=example.text, speaker=example.speaker.name))
    _save_partial = lambda i, tags, *args: _save(
        destination, ['index=%d' % i] + tags, *args, obscure=obscure)

    # Save the target predictions
    if has_target_audio:
        # NOTE: Adds `spectrogram_audio` and `spectrogram` column.
        dataset = add_spectrogram_column(dataset, on_disk=False)
        for i, example in zip(indicies, dataset):
            if example.spectrogram_audio is not None:
                waveform = example.spectrogram_audio.cpu().numpy()
                audio_path = _save_partial(i, ['type=gold'], example.speaker, waveform)
                add_to_metadata(example, audio_path=audio_path, index=i, type='gold')
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
        for i, example in zip(indicies, dataset):
            waveform = griffin_lim(example.predicted_spectrogram.cpu().numpy())
            audio_path = _save_partial(i, ['type=griffin_lim'], example.speaker, waveform)
            add_to_metadata(example, audio_path=audio_path, index=i, type='griffin_lim')
    else:
        logger.info('Skipping the writing of griffin-lim predictions.')

    # Save the signal model predictions
    if signal_model_checkpoint is not None and (has_target_audio or
                                                spectrogram_model_checkpoint is not None):
        logger.info('The signal model path is: %s', signal_model_checkpoint.path)
        signal_model_model = signal_model_checkpoint.model.to_inferrer()
        use_predicted = spectrogram_model_checkpoint is not None

        # NOTE: Sort by spectrogram lengths to batch similar sized outputs together
        iterator = list(zip(dataset, indicies))
        get_length = lambda e: getattr(e[0], 'predicted_spectrogram'
                                       if use_predicted else 'spectrogram').shape[0]
        iterator = sorted(iterator, key=get_length, reverse=True)

        for example, i in iterator:
            example = tensors_to(example, device=torch.device('cpu'), non_blocking=True)
            spectrogram = example.predicted_spectrogram if use_predicted else example.spectrogram
            logger.info('Predicting signal from spectrogram of size %s.', spectrogram.shape)
            with evaluate(signal_model_model):
                # [local_length, local_features_size] → [signal_length]
                predicted_coarse, predicted_fine, _ = signal_model_model(spectrogram)
                waveform = combine_signal(predicted_coarse, predicted_fine, return_int=True)
                waveform = waveform.cpu().numpy()

            audio_path = _save_partial(i, ['type=wave_rnn'], example.speaker, waveform)
            add_to_metadata(example, audio_path=audio_path, index=i, type='wave_rnn')
            logger.info('-' * 100)
    else:
        logger.info('Skipping the writing of neural vocoder predictions.')

    pandas.DataFrame(metadata).to_csv(str((destination / metadata_filename)), index=False)


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
    parser.add_argument(
        '-n', '--num_samples', type=int, help='Number of samples to generate in total.', default=50)
    args = parser.parse_args()

    # NOTE: Load early and crash early by ensuring that the checkpoint exists and is not corrupt.
    if args.signal_model is not None:
        args.signal_model = Checkpoint.from_path(args.signal_model)

    dataset = _get_dev_dataset
    if args.spectrogram_model is not None:
        args.spectrogram_model = Checkpoint.from_path(args.spectrogram_model)
        if args.text is not None:
            speakers = args.spectrogram_model.input_encoder.speaker_encoder.speakers
            dataset = [TextSpeechRow(t, s, None) for t, s in product(args.text, speakers)]
    elif args.text is not None:
        raise argparse.ArgumentTypeError(
            'For custom data, `--spectrogram_model` must be defined; '
            'otherwise, there is no accompanying audio data for evaluation.')

    set_hparams()

    main(
        dataset=dataset,
        spectrogram_model_checkpoint=args.spectrogram_model,
        signal_model_checkpoint=args.signal_model,
        num_samples=args.num_samples if args.text is None else len(dataset))

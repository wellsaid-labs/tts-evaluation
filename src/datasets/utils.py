from collections import Counter
from functools import partial
from pathlib import Path

import logging
import pathlib
import pprint

import numpy
import pandas
import torch

from src.audio import get_log_mel_spectrogram
from src.audio import normalize_audio
from src.audio import read_audio
from src.datasets.constants import TextSpeechRow
from src.distributed import download_file_maybe_extract
from src.hparams import configurable
from src.hparams import ConfiguredArg
from src.utils import batch_predict_spectrograms
from src.utils import Checkpoint
from src.utils import OnDiskTensor
from src.utils import ROOT_PATH

import src.distributed

logger = logging.getLogger(__name__)
pprint = pprint.PrettyPrinter(indent=4)


@configurable
def add_predicted_spectrogram_column(data,
                                     checkpoint_path,
                                     device,
                                     batch_size=ConfiguredArg(),
                                     on_disk=True,
                                     aligned=True,
                                     use_tqdm=True):
    """ For each example in ``data``, add predicted spectrogram data.

    Args:
        data (iterable of TextSpeechRow)
        checkpoint_path (src or Path): Path to checkpoint for the spectrogram model.
        device (torch.device): Device to run prediction on.
        batch_size (int, optional)
        on_disk (bool, optional): Save the tensor to disk returning a ``OnDiskTensor`` instead of
            ``torch.Tensor``. Furthermore, this utilizes the disk as a cache.
        aligned (bool, optional): If ``True`` predict a ground truth aligned spectrogram.
        use_tqdm (bool, optional): Write a progress bar to standard streams.

    Returns:
        (iterable of TextSpeechRow)
    """
    logger.info('Adding a predicted spectrogram column to dataset.')
    checkpoint = Checkpoint.from_path(checkpoint_path, device=device, load_random_state=False)

    if aligned and not all([r.spectrogram is not None for r in data]):
        raise RuntimeError("Spectrogram column of ``TextSpeechRow`` must not be ``None``.")

    filenames = None
    if on_disk:
        # Create unique paths for saving to disk
        model_name = str(checkpoint.path.resolve().relative_to(ROOT_PATH))
        model_name = model_name.replace('/', '_').replace('.', '_')

        def to_filename(example):
            if example.audio_path is None:
                parent = pathlib.Path('/tmp')
                # Learn more:
                # https://computinglife.wordpress.com/2008/11/20/why-do-hash-functions-use-prime-numbers/
                name = 31 * hash(example.text) + 97 * hash(example.speaker)
            else:
                parent = example.audio_path.parent
                name = example.audio_path.stem
            return parent / 'predicted_spectrogram({},{},aligned={}).npy'.format(
                name, model_name, aligned)

        filenames = [to_filename(example) for example in data]
        if all([f.is_file() for f in filenames]):
            # TODO: Consider logging metrics on the predicted spectrogram despite the
            # predictions being cached.
            tensors = [OnDiskTensor(f) for f in filenames]
            return [e._replace(predicted_spectrogram=t) for e, t in zip(data, tensors)]

    tensors = batch_predict_spectrograms(
        data,
        checkpoint.input_encoder,
        checkpoint.model,
        device,
        batch_size,
        filenames=filenames,
        aligned=aligned,
        use_tqdm=use_tqdm)
    return [e._replace(predicted_spectrogram=t) for e, t in zip(data, tensors)]


def _add_spectrogram_column(example, on_disk=True):
    """ Adds spectrogram to ``example``.

    Args:
        example (TextSpeechRow): Example of text and speech.
        on_disk (bool, optional): Save the tensor to disk returning a ``OnDiskTensor`` instead of
            ``torch.Tensor``.

    Returns:
        (TextSpeechRow): Row of text and speech aligned data with spectrogram data.
    """
    import librosa

    audio_path = example.audio_path

    if on_disk:
        spectrogram_audio_path = audio_path.parent / 'pad({}).npy'.format(audio_path.stem)
        spectrogram_path = audio_path.parent / 'spectrogram({}).npy'.format(audio_path.stem)
        is_cached = spectrogram_path.is_file() and spectrogram_audio_path.is_file()

    # For the distributed case, allow only the master node to save to disk while the worker nodes
    # optimistically assume the file already exists.

    if not on_disk or (on_disk and not is_cached and src.distributed.is_master()):
        # Compute and save to disk the spectrogram and audio
        assert audio_path.is_file(), 'Audio path must be a file %s' % audio_path
        signal = read_audio(audio_path)
        signal = librosa.effects.trim(signal)[0]
        log_mel_spectrogram, padding = get_log_mel_spectrogram(signal)
        log_mel_spectrogram = torch.from_numpy(log_mel_spectrogram)

        # Pad so: ``log_mel_spectrogram.shape[0] % signal.shape[0] == frame_hop``
        # This property is required for the vocoder.
        padded_signal = numpy.pad(signal, padding, mode='constant', constant_values=0)
        padded_signal = torch.from_numpy(padded_signal)

        if on_disk:
            return example._replace(
                spectrogram_audio=OnDiskTensor.from_tensor(spectrogram_audio_path, padded_signal),
                spectrogram=OnDiskTensor.from_tensor(spectrogram_path, log_mel_spectrogram))

        return example._replace(spectrogram_audio=padded_signal, spectrogram=log_mel_spectrogram)

    return example._replace(
        spectrogram_audio=OnDiskTensor(spectrogram_audio_path),
        spectrogram=OnDiskTensor(spectrogram_path))


def add_spectrogram_column(data, on_disk=True):
    """ For each example in ``data``, add spectrogram data.

    Args:
        data (iterables of TextSpeechRow)
        on_disk (bool, optional): Save the tensor to disk returning a ``OnDiskTensor`` instead of
            ``torch.Tensor``.

    Returns:
        (iterable of TextSpeechRow): Iterable of text speech rows along with spectrogram
            data.
    """
    logger.info('Adding a spectrogram column to dataset.')
    return src.distributed.map_multiprocess(data, partial(_add_spectrogram_column, on_disk=on_disk))


def _normalize_audio_column_helper(example):
    return example._replace(audio_path=normalize_audio(example.audio_path))


def normalize_audio_column(data):
    """ For each example in ``data``, normalize the audio using SoX and update the ``audio_path``.

    Args:
        data (iterables of TextSpeechRow)

    Returns:
        (iterable of TextSpeechRow): Iterable of text speech rows with ``audio_path`` updated.
    """
    return src.distributed.map_multiprocess(data, _normalize_audio_column_helper)


def filter_(function, dataset):
    """ Similar to the python ``filter`` function with extra logging.

    Args:
        function (callable)
        dataset (iterable of TextSpeechRow)

    Returns:
        iterable of TextSpeechRow
    """
    positive = list(filter(function, dataset))
    negative = list(filter(lambda e: not function(e), dataset))
    speaker_distribution = Counter([example.speaker for example in negative])
    logger.info('Filtered out %d examples via ``%s`` with a speaker distribution of:\n%s',
                len(negative), function.__name__, pprint.pformat(speaker_distribution))
    return positive


def _dataset_loader(
        extracted_name,
        url,
        speaker,
        url_filename=None,
        check_files=['{metadata_filename}'],
        directory=ROOT_PATH / 'data',
        metadata_filename='{directory}/{extracted_name}/metadata.csv',
        metadata_text_column='Content',
        metadata_audio_column='WAV Filename',
        metadata_audio_path='{directory}/{extracted_name}/wavs/{metadata_audio_column_value}',
        **kwargs):
    """ Load a standard speech dataset.

    A standard speech dataset has these invariants:
        - The file structure is similar to:
            dataset/
                metadata.csv
                wavs/
                    audio1.wav
                    audio2.wav
        - The metadata CSV file contains a mapping of audio transcriptions to audio filenames.
        - The dataset contains one speaker.
        - The dataset is stored in a ``tar`` or ``zip`` at some url.

    Args:
        extracted_name (str): Name of the extracted dataset directory.
        url (str): URL of the dataset file.
        speaker (src.datasets.Speaker): The dataset speaker.
        check_files (list of str, optional): The download is considered successful, if these files
            exist.
        url_filename (str, optional): Name of the file downloaded; Otherwise, a filename is
            extracted from the url.
        directory (str or Path, optional): Directory to cache the dataset.
        metadata_filename (str, optional): The filename for the metadata file.
        metadata_text_column (str, optional): Column name or index with the audio transcript.
        metadata_audio_column (str, optional): Column name or index with the audio filename.
        metadata_audio_path (str, optional): String template for the audio path given the
            ``metadata_audio_column`` value.
        **kwargs: Key word arguments passed to ``pandas.read_csv``.

    Returns:
        list of TextSpeechRow: Dataset with audio filenames and text annotations.
    """
    logger.info('Loading %s speech dataset', extracted_name)
    directory = Path(directory)
    metadata_filename = metadata_filename.format(directory=directory, extracted_name=extracted_name)
    check_files = [
        str(Path(f.format(metadata_filename=metadata_filename)).absolute()) for f in check_files
    ]
    download_file_maybe_extract(
        url=url,
        directory=str(directory.absolute()),
        check_files=check_files,
        filename=url_filename)
    dataframe = pandas.read_csv(Path(metadata_filename), **kwargs)
    return [
        TextSpeechRow(
            text=row[metadata_text_column].strip(),
            audio_path=Path(
                metadata_audio_path.format(
                    directory=directory,
                    extracted_name=extracted_name,
                    metadata_audio_column_value=row[metadata_audio_column])),
            speaker=speaker,
            metadata={
                k: v
                for k, v in row.items()
                if k not in [metadata_text_column, metadata_audio_column]
            })
        for _, row in dataframe.iterrows()
    ]

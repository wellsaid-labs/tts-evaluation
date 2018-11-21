from torch.multiprocessing import Pool

import logging
import os

from torchnlp.utils import shuffle as do_deterministic_shuffle
from tqdm import tqdm

import librosa
import numpy
import torch

from src.audio import get_log_mel_spectrogram
from src.audio import read_audio

logger = logging.getLogger(__name__)


def normalize_audio(audio_path,
                    resample=24000,
                    norm=False,
                    guard=True,
                    lower_hertz=None,
                    upper_hertz=None,
                    loudness=False):
    """ Normalize audio with the SoX library and cache processed audio.

    Args:
        audio_path (Path): Path to a audio file.
        resample (int or None, optional): If integer is provided, uses SoX to create resampled
            files.
        norm (bool, optional): Automatically invoke the gain effect to guard against clipping and to
            normalise the audio.
        guard (bool, optional): Automatically invoke the gain effect to guard against clipping.
        lower_hertz (int, optional): Apply a sinc kaiser-windowed high-pass.
        upper_hertz (int, optional): Apply a sinc kaiser-windowed low-pass.
        loudness (bool, optioanl): Normalize the subjective perception of loudness level based on
            ISO 226.

    Returns:
        (str): Filename of the processed file.
    """
    lower_hertz = str(lower_hertz) if lower_hertz is not None else ''
    upper_hertz = str(upper_hertz) if upper_hertz is not None else ''

    # Create cach'd filename
    stem = audio_path.stem
    # HACK: -.001 DB applied to prevent clipping warning.
    stem = ('norm(%s,-.001)' % stem) if norm else stem
    stem = ('guard(%s)' % stem) if guard else stem
    stem = ('rate(%s,%d)' % (stem, resample)) if resample is not None else stem
    stem = (
        'sinc(%s,%s,%s)' % (stem, lower_hertz, upper_hertz)) if lower_hertz or upper_hertz else stem
    stem = ('loudness(%s)' % stem) if loudness else stem

    dest_path = audio_path.parent / '{}{}'.format(stem, audio_path.suffix)
    if stem == audio_path.stem or dest_path.is_file():
        return dest_path

    norm_flag = '--norm=-.001' if norm else ''
    guard_flag = '--guard' if guard else ''
    sinc_command = 'sinc %s-%s' % (lower_hertz, upper_hertz) if lower_hertz or upper_hertz else ''
    loudness_command = 'loudness' if loudness else ''
    resample_command = 'rate %s' % (resample if resample is not None else '',)
    commands = ' '.join([resample_command, sinc_command, loudness_command])
    flags = ' '.join([norm_flag, guard_flag])
    command = 'sox "%s" %s "%s" %s ' % (audio_path, flags, dest_path, commands)
    os.system(command)

    return dest_path


def _predict_spectrogram(checkpoint, text, speaker, real_spectrogram):
    """ Predict a ground truth aligned spectrogram.

    Args:
        checkpoint (Checkpoint): Checkpoint for the spectrogram model.
        text (str)
        speaker (src.datasets.Speaker)
        real_spectrogram (numpy.float32 [num_frames, frame_channels])

    Returns:
        (torch.FloatTensor [num_frames, frame_channels]): Predicted spectrogram.
    """
    with torch.no_grad():
        checkpoint.model.train(mode=False)

        real_spectrogram = torch.from_numpy(real_spectrogram)

        # [num_tokens]
        encoded_text = checkpoint.text_encoder.encode(text)
        encoded_speaker = checkpoint.speaker_encoder.encode(speaker)

        # [num_frames, frame_channels]
        predicted_spectrogram = checkpoint.model(
            encoded_text, encoded_speaker, ground_truth_frames=real_spectrogram)[1]
    return predicted_spectrogram


def compute_spectrogram(audio_path, text=None, speaker=None, spectrogram_model_checkpoint=None):
    """ Computes the spectrogram and saves to cache returning the cache filename.

    Args:
        audio_path (Path): Path to a audio file.
        text (str, optional): Text used to compute a predicted spectrogram.
        speaker (src.datasets.Speaker, optional): Speaker used to compute a predicted spectrogram.
        spectrogram_model_checkpoint (Checkpoint, optional): Spectrogram model checkpoint to
            compute a predicted spectrogram in addition to the real spectrogram.

    Returns:
        (Path): Filename of cache'd spectrogram file.
        (Path): Filename of cache'd spectrogram aligned audio file.
        (Path or None): Filename of cache'd predicted spectrogram audio file.
    """
    include_predicted = (
        text is not None and speaker is not None and spectrogram_model_checkpoint is not None)

    dest_padded_audio = audio_path.parent / 'pad({}).npy'.format(audio_path.stem, audio_path.suffix)
    dest_spectrogram = audio_path.parent / 'spectrogram({}).npy'.format(audio_path.stem)
    dest_predicted_spectrogram = None

    if include_predicted:
        spectrogram_model_name = str(spectrogram_model_checkpoint.path).replace('/', '_').replace(
            '.', '_')
        dest_predicted_spectrogram = 'predicted_spectrogram({},{}).npy'.format(
            audio_path.stem, spectrogram_model_name)
        dest_predicted_spectrogram = audio_path.parent / dest_predicted_spectrogram

    if dest_spectrogram.is_file() and dest_padded_audio.is_file() and (
            dest_predicted_spectrogram is None or dest_predicted_spectrogram.is_file()):
        return dest_padded_audio, dest_spectrogram, dest_predicted_spectrogram

    if dest_spectrogram.is_file() and dest_padded_audio.is_file():
        log_mel_spectrogram = numpy.load(str(dest_spectrogram))
    else:
        signal = read_audio(audio_path)
        signal = librosa.effects.trim(signal)[0]
        log_mel_spectrogram, padding = get_log_mel_spectrogram(signal)

        # Pad so: ``log_mel_spectrogram.shape[0] % signal.shape[0] == frame_hop``
        # This property is required for the vocoder.
        padded_signal = numpy.pad(signal, padding, mode='constant', constant_values=0)
        numpy.save(str(dest_padded_audio), padded_signal, allow_pickle=False)
        numpy.save(str(dest_spectrogram), log_mel_spectrogram, allow_pickle=False)

    if include_predicted:
        predicted_spectrogram = _predict_spectrogram(spectrogram_model_checkpoint, text, speaker,
                                                     log_mel_spectrogram)
        numpy.save(str(dest_predicted_spectrogram), predicted_spectrogram, allow_pickle=False)

    return dest_padded_audio, dest_spectrogram, dest_predicted_spectrogram


def split_dataset(dataset, splits, random_seed=123):
    """
    Args:
        dataset (list): Dataset to split.
        splits (tuple): Tuple of percentages determining dataset splits.
        random_seed (int, optional): Shuffle deterministically provided some random seed always
            returning the same split given the same dataset.

    Returns:
        (list): splits of the dataset

    Example:
        >>> dataset = [1, 2, 3, 4, 5]
        >>> splits = (.6, .2, .2)
        >>> split_dataset(dataset, splits, random_seed=123)
        [[4, 2, 5], [3], [1]]
    """
    if random_seed is not None:
        do_deterministic_shuffle(dataset, random_seed=random_seed)

    assert sum(splits) == 1, 'Splits must sum to 100%'
    splits = [round(s * len(dataset)) for s in splits]
    datasets = []
    for split in splits[:-1]:
        datasets.append(dataset[:split])
        dataset = dataset[split:]
    datasets.append(dataset)
    return datasets


def process_with_processes(data, processing_func):
    """ Process ``data`` with ``processing_func`` using threads and ``tqdm``.

    Args:
        data (iterable)
        processing_func (callable)

    Returns:
        processed (list)
    """
    logger.info('Processing dataset...')
    data = list(data)
    with Pool() as pool:
        processed = list(tqdm(pool.imap(processing_func, data), total=len(data)))
    return processed

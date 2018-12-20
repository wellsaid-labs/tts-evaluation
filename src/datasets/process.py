from functools import partial
from math import inf

import logging
import os

from torch.multiprocessing import Pool
from torch.nn.functional import mse_loss
from torchnlp.utils import shuffle as do_deterministic_shuffle
from tqdm import tqdm

import librosa
import numpy
import torch
import torchnlp.download

from src.audio import get_log_mel_spectrogram
from src.audio import read_audio
from src.datasets.constants import SpectrogramTextSpeechRow
from src.hparams import configurable
from src.utils import Checkpoint
from src.utils import collate_sequences
from src.utils import DataLoader
from src.utils import evaluate
from src.utils import get_average_norm
from src.utils import get_weighted_stdev
from src.utils import lengths_to_mask
from src.utils import OnDiskTensor
from src.utils import ROOT_PATH
from src.utils import sort_by_spectrogram_length
from src.utils import tensors_to
from src.visualize import AccumulatedMetrics

import src.distributed

logger = logging.getLogger(__name__)

# LEARN MORE: https://github.com/pytorch/pytorch/issues/973
torch.multiprocessing.set_sharing_strategy('file_system')


def _download_file_maybe_extract(*args, **kwargs):
    """
    Alias to ``torchnlp.download.download_file_maybe_extract`` that considers the distributed
    environment.
    """
    if not src.distributed.in_use() or src.distributed.is_master():
        torchnlp.download.download_file_maybe_extract(*args, **kwargs)

    # Ensure data is downloaded before both worker and master proceed.
    if src.distributed.in_use():
        torch.distributed.barrier()


def _split_dataset(dataset, splits, random_seed=123):
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
        >>> _split_dataset(dataset, splits, random_seed=123)
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


def _process_in_parallel(data, processing_func, use_tqdm=True):
    """ Process ``data`` with ``processing_func`` using threads and ``tqdm``.

    Args:
        data (iterable)
        processing_func (callable)
        use_tqdm (bool): Attach a progress bar to processing.

    Returns:
        processed (list)
    """
    data = list(data)

    use_pool = not src.distributed.in_use() or src.distributed.is_master()
    if use_pool:
        pool = Pool()
        iterator = pool.imap(processing_func, data)
    else:  # PyTorch workers should not expect to do serious work
        iterator = (processing_func(row) for row in data)

    if use_tqdm:
        iterator = tqdm(iterator, total=len(data))
    processed = list(iterator)

    if use_pool:  # Ensure pool work is finished
        pool.close()
        pool.join()

    # Ensure data is processed before both worker and master proceed.
    if src.distributed.in_use():
        torch.distributed.barrier()

    return processed


def _normalize_audio_and_cache(audio_path,
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

    # Allow only the master node to save to disk while the worker nodes optimistically assume
    # the file already exists.
    if src.distributed.in_use() and not src.distributed.is_master():
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


def _load_fn(row, text_encoder, speaker_encoder):
    """ Load function for loading a single row.

    Args:
        row (SpectrogramTextSpeechRow)
        text_encoder (torchnlp.TextEncoder): Text encoder used to encode and decode the text.
        speaker_encoder (torchnlp.TextEncoder): Text encoder used to encode and decode the speaker.

    Returns:
        (SpectrogramTextSpeechRow)
    """
    return row._replace(
        text=text_encoder.encode(row.text),
        speaker=speaker_encoder.encode(row.speaker),
        spectrogram=(row.spectrogram.to_tensor()
                     if isinstance(row.spectrogram, OnDiskTensor) else row.spectrogram),
        spectrogram_audio=(row.spectrogram_audio.to_tensor() if isinstance(
            row.spectrogram_audio, OnDiskTensor) else row.spectrogram_audio))


def _predict_spectrogram(data,
                         checkpoint_path,
                         device,
                         batch_size,
                         aligned=True,
                         on_disk=True,
                         use_tqdm=True):
    """ Predict spectrograms from a list of ``SpectrogramTextSpeechRow`` rows and maybe cache.

    Args:
        data (iterable of SpectrogramTextSpeechRow)
        checkpoint_path (src or Path): Path to checkpoint for the spectrogram model.
        device (torch.device): Device to run prediction on.
        batch_size (int)
        aligned (bool): If ``True``, predict a ground truth aligned spectrogram.
        on_disk (bool, optional): Save the tensor to disk, returning a ``OnDiskTensor`` instead of
            ``torch.Tensor``.
        use_tqdm (bool): Write a progress bar to console.

    Returns:
        (iterable of SpectrogramTextSpeechRow)
    """
    checkpoint = Checkpoint.from_path(checkpoint_path, device=device)
    model_name = str(checkpoint.path.resolve().relative_to(ROOT_PATH))
    model_name = model_name.replace('/', '_').replace('.', '_')

    return_ = []
    for row in data:
        destination = 'predicted_spectrogram({},{},aligned={}).npy'.format(
            row.audio_path.stem, model_name, aligned)
        on_disk_tensor = OnDiskTensor(row.audio_path.parent / destination)
        return_.append(row._replace(predicted_spectrogram=on_disk_tensor))

    is_cached = all([r.predicted_spectrogram.does_exist() for r in return_])
    if not is_cached and (not src.distributed.in_use() or src.distributed.is_master()):
        return_ = sort_by_spectrogram_length(return_)
        text_encoder = checkpoint.text_encoder
        speaker_encoder = checkpoint.speaker_encoder
        loader = DataLoader(
            return_,
            batch_size=batch_size,
            load_fn=partial(_load_fn, text_encoder=text_encoder, speaker_encoder=speaker_encoder),
            post_processing_fn=partial(tensors_to, device=device, non_blocking=True),
            collate_fn=partial(collate_sequences, dim=1),
            pin_memory=True,
            use_tqdm=use_tqdm)
        with evaluate(checkpoint.model, device=device):
            metrics = AccumulatedMetrics()
            for batch in loader:
                # Predict spectrogram
                text, speaker = batch.text[0], batch.speaker[0]
                spectrogram, lengths = batch.spectrogram
                if aligned:
                    _, predicted, _, alignments = checkpoint.model(text, speaker, spectrogram)
                else:
                    _, predicted, _, alignments, lengths = checkpoint.model(text, speaker)

                # Compute some metrics for logging
                mask = lengths_to_mask(lengths, device=predicted.device).transpose(0, 1)
                metrics.add_multiple_metrics({
                    'attention_norm': get_average_norm(alignments, norm=inf, dim=2, mask=mask),
                    'attention_std': get_weighted_stdev(alignments, dim=2, mask=mask),
                }, mask.sum())
                if aligned:
                    mask = mask.unsqueeze(2).expand_as(predicted)
                    loss = mse_loss(predicted, spectrogram, reduction='none')
                    metrics.add_metric('loss', loss.masked_select(mask).mean(), mask.sum())

                # Save to disk
                splits = predicted.split(1, dim=1)
                iterator = zip(splits, lengths, batch.predicted_spectrogram)
                for tensor, length, on_disk_tensor in iterator:
                    # split [num_frames, 1, frame_channels]
                    on_disk_tensor.from_tensor(tensor[:length, 0])

            metrics.log_epoch_end(lambda k, v: logger.info('Prediction metric (%s): %s', k, v))

    if on_disk:
        return return_

    # NOTE: This should be quick because the saved tensor should still be in memory
    return [r._replace(predicted_spectrogram=r.predicted_spectrogram.to_tensor()) for r in return_]


def _compute_spectrogram(row, on_disk=True):
    """ Computes the spectrogram and maybe caches.

    Args:
        row (TextSpeechRow): Row of text and speech aligned data.
        on_disk (bool, optional): Save the tensor to disk, returning a ``OnDiskTensor`` instead of
            ``torch.Tensor``.

    Returns:
        (SpectrogramTextSpeechRow): Row of text and speech aligned data with spectrogram data.
    """
    audio_path = row.audio_path
    dest_padded_audio = audio_path.parent / 'pad({}).npy'.format(audio_path.stem, audio_path.suffix)
    dest_spectrogram = audio_path.parent / 'spectrogram({}).npy'.format(audio_path.stem)
    return_ = SpectrogramTextSpeechRow(
        **row._asdict(),
        spectrogram_audio=OnDiskTensor(dest_padded_audio),
        spectrogram=OnDiskTensor(dest_spectrogram),
        predicted_spectrogram=None)

    # For the distributed case, allow only the master node to save to disk while the worker nodes
    # optimistically assume the file already exists.
    is_cached = dest_spectrogram.is_file() and dest_padded_audio.is_file()
    if (not is_cached and (not src.distributed.in_use() or src.distributed.is_master())):
        # Compute and save to disk the spectrogram and audio
        assert audio_path.is_file(), 'Audio path must be a file %s' % audio_path
        signal = read_audio(audio_path)
        signal = librosa.effects.trim(signal)[0]
        log_mel_spectrogram, padding = get_log_mel_spectrogram(signal)

        # Pad so: ``log_mel_spectrogram.shape[0] % signal.shape[0] == frame_hop``
        # This property is required for the vocoder.
        padded_signal = numpy.pad(signal, padding, mode='constant', constant_values=0)

        return_.spectrogram_audio.from_tensor(padded_signal)
        return_.spectrogram.from_tensor(log_mel_spectrogram)

    if not on_disk:
        # NOTE: This should be quick because the saved tensor should still be in memory
        return_ = return_._replace(
            spectrogram_audio=return_.spectrogram_audio.to_tensor(),
            spectrogram=return_.spectrogram.to_tensor())

    return return_


@configurable
def compute_spectrograms(data, on_disk=True, checkpoint_path=None, **kwargs):
    """ Given rows of text speech rows, computes related spectrograms both real and predicted.

    Args:
        data (iterables of TextSpeechRow)
        on_disk (bool, optional): Save the tensor to disk, returning a ``OnDiskTensor`` instead of
            ``torch.Tensor``.
        checkpoint_path (str or None, optional): Spectrogram model to predict a ground truth aligned
            spectrogram.
        **kwargs: Additional arguments passed to ``_predict_spectrogram``.

    Returns:
        (iterable of SpectrogramTextSpeechRow): Iterable of text speech rows along with spectrogram
            data.
    """
    logger.info('Computing spectrograms...')
    data = _process_in_parallel(data, partial(_compute_spectrogram, on_disk=on_disk))
    if checkpoint_path is not None:
        logger.info('Predicting spectrograms...')
        data = _predict_spectrogram(data, checkpoint_path, on_disk=on_disk, **kwargs)

    # Ensure data is processed before both worker and master proceed.
    if src.distributed.in_use():
        torch.distributed.barrier()

    return data

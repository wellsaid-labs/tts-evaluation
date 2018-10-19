from concurrent.futures import ThreadPoolExecutor
from torch.multiprocessing import cpu_count

import logging
import os

from torchnlp.datasets import Dataset
from tqdm import tqdm

from src.utils import split_dataset

logger = logging.getLogger(__name__)


def process_audio(wav,
                  resample=24000,
                  norm=False,
                  guard=True,
                  lower_hertz=None,
                  upper_hertz=None,
                  loudness=False):
    """ Process audio with the SoX library.

    Args:
        wav (Path): Path to a audio file.
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

    name = wav.name
    if resample is not None:
        name = name.replace('.wav', '-rate=%d.wav' % resample)
    if norm:
        name = name.replace('.wav', '-norm=-.001.wav')
    if loudness:
        name = name.replace('.wav', '-loudness.wav')
    if guard:
        name = name.replace('.wav', '-guard.wav')
    if lower_hertz or upper_hertz:
        name = name.replace('.wav', '-sinc_%s_%s.wav' % (lower_hertz, upper_hertz))

    destination = wav.parent / name
    if name == wav.name or destination.is_file():
        return destination

    # NOTE: -.001 DB applied to prevent clipping.
    norm_flag = '--norm=-.001' if norm else ''
    guard_flag = '--guard' if guard else ''
    sinc_command = 'sinc %s-%s' % (lower_hertz, upper_hertz) if lower_hertz or upper_hertz else ''
    loudness_command = 'loudness' if loudness else ''
    resample_command = 'rate %s' % (resample if resample is not None else '',)
    commands = ' '.join([resample_command, sinc_command, loudness_command])
    flags = ' '.join([norm_flag, guard_flag])
    command = 'sox "%s" %s "%s" %s ' % (wav, flags, destination, commands)

    os.system(command)
    return destination


def process_all(extract_fun, data, splits, random_seed, deterministic_shuffle=True):
    """
    Given a generator yielding a pair of ``(text, audio_filename)``, run the ``extract_fun``
    function which processes text and audio, then split the resulting data.

    Args:
        extract_fun (callable): The extract function of type
            :class:`Callable[Tuple[str, str], dict]`.
        data (callable): The generator yielding (text, audio_filename) pairs.
        splits (tuple): The number of splits and cardinality of dataset splits.
        random_seed (int): Random seed used to determine the splits.
        deterministic_shuffle (bool, optional): If ``True`` determinisitically shuffle the dataset
            before splitting.

    Returns:
        A tuple of train/dev :class:`torchnlp.datasets.Dataset`.
    """
    data = list(data)
    with ThreadPoolExecutor(cpu_count()) as e:
        examples = list(tqdm(e.map(extract_fun, data), total=len(data)))

    splits = split_dataset(
        examples,
        splits=splits,
        deterministic_shuffle=deterministic_shuffle,
        random_seed=random_seed)
    return tuple(Dataset(split) for split in splits)

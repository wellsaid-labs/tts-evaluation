import os
import logging
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count

from torchnlp.datasets import Dataset
from tqdm import tqdm

from src.utils import split_dataset

logger = logging.getLogger(__name__)


def read_speech_data(metadata_path, audio_directory='wavs', text_column=1, check_wavfiles=True):
    """
    Given the path to a metadata.csv file, yield pairs of (text, wav_filename).

    Args:
        metadata_path: pathlib.Path is the location of the metadata.csv file.
        audio_directory: str is the name of the directory containing the wav files.
        text_column: 0-indexed column to extract text from.
        check_wavfiles: If False, skip the check for existence of wav files.

    Returns:
        A generator yielding pairs of (text, wav_filename), where wav_filename is a Path object.
    """
    lines = list(metadata_path.open())
    missing = 0
    for line in lines:
        columns = line.split('|')
        wav_filename = metadata_path.parent / audio_directory / (columns[0] + '.wav')
        if wav_filename.exists() or not check_wavfiles:
            yield columns[text_column], wav_filename
        else:
            logger.warning("\n{} is missing".format(wav_filename))
            missing += 1

    if check_wavfiles and missing > 0:
        logger.warning("{} missing wav files".format(missing))


def process_audio(wav,
                  resample=24000,
                  norm=True,
                  guard=True,
                  lower_hertz=0,
                  upper_hertz=12000,
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
    command = 'sox %s %s %s %s ' % (wav, flags, str(destination), commands)

    os.system(command)
    return destination


def process_all(extract_fun, data, splits, random_seed, check_wavfiles=True):
    """
    Given a generator yielding a pair of (text, wav_filename), run the extract_fun function
    which processes text and audio, then split the resulting data into train/dev.

    Args:
        extract_fun: The extract function of type :class:`Callable[Tuple[str, str], dict]`.
        data: The generator yielding (text, wav_filename) pairs.
        random_seed: Random seed used to determine the splits.
        splits (tuple): The number of splits and cardinality of dataset splits.
        check_wavfiles: If False, skip the check for existence of wav files.

    Returns:
        A tuple of train/dev :class:`torchnlp.datasets.Dataset`.
    """
    data = list(data)

    # NOTE: This is causing test_lj_speech_dataset_audio_processing to fail intermittently!!!
    # Need to look into fixing that test.
    with ThreadPoolExecutor(cpu_count()) as e:
        examples = list(tqdm(e.map(extract_fun, data), total=len(data)))

    if check_wavfiles:
        for example in examples:
            if not os.path.isfile(example['wav_filename']):
                logger.error("Processed audio file {} missing!".format(example['wav_filename']))

    splits = split_dataset(
        examples, splits=splits, deterministic_shuffle=True, random_seed=random_seed)
    return tuple(Dataset(split) for split in splits)

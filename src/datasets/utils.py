from collections import Counter
from functools import partial
from itertools import groupby
from pathlib import Path

import logging
import os
import pprint
import string

from hparams import configurable
from hparams import HParam
from multiprocessing.pool import ThreadPool
from phonemizer.phonemize import phonemize
from phonemizer.separator import Separator
from third_party import LazyLoader
from torchnlp.download import download_file_maybe_extract
from tqdm import tqdm

import hparams
import torch

librosa = LazyLoader('librosa', globals(), 'librosa')
pandas = LazyLoader('pandas', globals(), 'pandas')

from src.audio import cache_get_audio_metadata
from src.audio import get_signal_to_db_mel_spectrogram
from src.audio import integer_to_floating_point_pcm
from src.audio import normalize_audio
from src.audio import pad_remainder
from src.audio import read_audio
from src.datasets.constants import TextSpeechRow
from src.environment import DATA_PATH
from src.environment import ROOT_PATH
from src.environment import TEMP_PATH
from src.environment import TTS_DISK_CACHE_NAME
from src.utils import batch_predict_spectrograms
from src.utils import OnDiskTensor
from src.utils import Pool

logger = logging.getLogger(__name__)
pprint = pprint.PrettyPrinter(indent=4)


@configurable
def add_predicted_spectrogram_column(data,
                                     checkpoint,
                                     device,
                                     batch_size=HParam(),
                                     on_disk=True,
                                     aligned=True,
                                     use_tqdm=True):
    """ For each example in ``data``, add predicted spectrogram data.

    Args:
        data (iterable of TextSpeechRow)
        checkpoint (src.utils.Checkpoint): Spectrogram model checkpoint.
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
    if aligned and not all([r.spectrogram is not None for r in data]):
        raise RuntimeError("Spectrogram column of ``TextSpeechRow`` must not be ``None``.")

    filenames = None
    if on_disk:
        # Create unique paths for saving to disk
        model_name = str(checkpoint.path.resolve().relative_to(ROOT_PATH))
        model_name = model_name.replace('/', '_').replace('.', '_')

        def to_filename(example):
            if example.audio_path is None:
                parent = TEMP_PATH
                # Learn more:
                # https://computinglife.wordpress.com/2008/11/20/why-do-hash-functions-use-prime-numbers/
                name = 31 * hash(example.text) + 97 * hash(example.speaker)
            else:
                parent = example.audio_path.parent
                name = example.audio_path.stem
            parent = parent if parent.name == TTS_DISK_CACHE_NAME else parent / TTS_DISK_CACHE_NAME
            parent.mkdir(exist_ok=True)
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
    audio_path = example.audio_path

    if on_disk:
        parent = audio_path.parent
        parent = parent if parent.name == TTS_DISK_CACHE_NAME else parent / TTS_DISK_CACHE_NAME
        spectrogram_audio_path = parent / 'pad({}).npy'.format(audio_path.stem)
        spectrogram_path = parent / 'spectrogram({}).npy'.format(audio_path.stem)
        is_cached = spectrogram_path.is_file() and spectrogram_audio_path.is_file()

    if not on_disk or (on_disk and not is_cached):
        # Compute and save to disk the spectrogram and audio
        assert audio_path.is_file(), 'Audio path must be a file %s' % audio_path
        signal = read_audio(audio_path)
        dtype = signal.dtype

        # TODO: The RMS function is a naive computation of loudness; therefore, it'd likely
        # be more accurate to use our spectrogram for trimming with augmentations like A-weighting.
        # TODO: The RMS function that trim uses mentions that it's likely better to use a
        # spectrogram if it's available:
        # https://librosa.github.io/librosa/generated/librosa.feature.rms.html?highlight=rms#librosa.feature.rms
        # TODO: `pad_remainder` could possibly add distortion if it's appended to non-zero samples;
        # therefore, it'd likely be beneficial to have a small fade-in and fade-out before
        # appending the zero samples.
        # TODO: We should consider padding more than just the remainder. We could additionally
        # pad a `frame_length` of padding so that further down the pipeline, any additional
        # padding does not affect the spectrogram due to overlap between the padding and the
        # real audio.
        signal = pad_remainder(signal)
        _, trim = librosa.effects.trim(integer_to_floating_point_pcm(signal))
        signal = signal[trim[0]:trim[1]]

        # TODO: Now that `get_signal_to_db_mel_spectrogram` is implemented in PyTorch, we could
        # batch process spectrograms. This would likely be faster. Also, it could be fast to
        # compute spectrograms on-demand.
        with torch.no_grad():
            db_mel_spectrogram = get_signal_to_db_mel_spectrogram()(
                torch.tensor(integer_to_floating_point_pcm(signal), requires_grad=False),
                aligned=True)
        assert signal.dtype == dtype, 'The signal `dtype` was changed.'
        signal = torch.tensor(signal, requires_grad=False)

        if on_disk:
            parent.mkdir(exist_ok=True)
            return example._replace(
                spectrogram_audio=OnDiskTensor.from_tensor(spectrogram_audio_path, signal),
                spectrogram=OnDiskTensor.from_tensor(spectrogram_path, db_mel_spectrogram))

        return example._replace(spectrogram_audio=signal, spectrogram=db_mel_spectrogram)

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
    partial_ = partial(_add_spectrogram_column, on_disk=on_disk)
    with ThreadPool(os.cpu_count()) as pool:
        # NOTE: `chunksize` with `imap` is more performant while allowing us to measure progress.
        # TODO: Consider using `imap_unordered` instead of `imap` because it is more performant,
        # learn more:
        # https://stackoverflow.com/questions/19063238/in-what-situation-do-we-need-to-use-multiprocessing-pool-imap-unordered
        # However, it's important to return the results in the same order as they came.
        return list(tqdm(pool.imap(partial_, data, chunksize=128), total=len(data)))


def _normalize_audio_column_helper(example, config):
    # TODO: Add a method for transfering global configuration between processes without private
    # variables.
    # TODO: After the global configuration is transfered, the functions need to be rechecked like
    # for a configuration, just in case the configuration is on a new process.
    hparams.hparams._configuration = config
    return example._replace(audio_path=normalize_audio(example.audio_path))


def normalize_audio_column(data):
    """ For each example in ``data``, normalize the audio using SoX and update the ``audio_path``.

    Args:
        data (iterables of TextSpeechRow)

    Returns:
        (iterable of TextSpeechRow): Iterable of text speech rows with ``audio_path`` updated.
    """
    cache_get_audio_metadata([e.audio_path for e in data])

    _normalize_audio_column_helper_partial = partial(
        _normalize_audio_column_helper, config=hparams.get_config())

    logger.info('Normalizing dataset audio using SoX.')
    with Pool() as pool:
        # NOTE: `chunksize` allows `imap` to be much more performant while allowing us to measure
        # progress.
        iterator = pool.imap(_normalize_audio_column_helper_partial, data, chunksize=1024)
        return_ = list(tqdm(iterator, total=len(data)))

    # NOTE: `cache_get_audio_metadata` for any new normalized audio paths.
    cache_get_audio_metadata([e.audio_path for e in return_])

    return return_


def normalize_text_clean_punctuation(dataset):
    """ Clean up texts missing spaces.

        In February 2020, it was discovered that many phrases were experiencing common typos
        throughout our datasets. The datasets should be cleaned before training, but in the interim
        this simple clean function will fix the following issues:

        ."Management
        ."Companies
        ."Critical
        ."Therefore
        ."Human
        ."More
        ."There

        .Other
        .People
        .Sameer
        .The
        .Value
        .There

    """

    typos = ['."', ' .', ' -', ' --']
    replacements = ['." ', '. ', ' - ', ' -- ']

    for row, example in enumerate(dataset):
        for t, r in zip(typos, replacements):
            i = example.text.find(t)
            if i > -1 and i + len(t) < len(
                    example.text) and example.text[i + len(t)] not in string.punctuation + ' \n':
                dataset[row] = example._replace(text=example.text.replace(t, r))

            if example.text[0] in ['.', '-'] and len(example.text) > 1:
                if example.text[1] not in string.punctuation + ' \n':
                    dataset[row] = example._replace(text=example.text[1:])
    return dataset


_separator_token = 'PHONE_SEPARATOR'
_separator = Separator(phone=_separator_token)


def phonemize_data(dataset, nlp):
    """
    Convert graphemes to phonemes in all rows in data.

    Args:
        dataset (iterable of TextSpeechRow)
        nlp (Language): spaCy Language object for tokenizing text

    Returns:
        dataset (iterable of TextSpeechRow)
    """

    assert _separator_token not in ''.join(
        r.text for r in dataset), 'WARNING! Phone separator is not unique.'

    logger.info('Phonemizing dataset...')

    list_of_tokenized_texts = [_phonemize_text(r.text, nlp=nlp) for r in dataset]

    phrase_pos = []
    to_phonemize = []
    for i, row in enumerate(list_of_tokenized_texts):
        for j, token in enumerate(row):
            if isinstance(token, tuple):
                pos = (i, j)
                phrase_pos.append(pos)
                to_phonemize.append(token[1])

    phonemes = _phonemize_phrases(to_phonemize)
    assert len(to_phonemize) == len(phonemes)

    for i, row in enumerate(list_of_tokenized_texts):
        for j, token in enumerate(row):
            if (i, j) in phrase_pos:
                list_of_tokenized_texts[i][j] = phonemes.pop(0)

    return [r._replace(text=''.join(l)) for r, l in zip(dataset, list_of_tokenized_texts)]


def _phonemize_text(text, nlp, pos_filter='PUNCT', **kwargs):
    """
    Tokenize text, phonemize phrases between punctuations, concatenate back together with
    punctuations, return tokenized representation of text with tuples ready to be converted
    to phonemes.

    Args:
        text (str): Text string from TextSpeechRow
        nlp (Language): spaCy Language object for tokenizing text
        pos_filter (str): spaCy part of speech filter
        language (str): Translation language rules to apply to phonemizer
        **kwargs: Utility arguments passed to phonemizer.phonemize

    Returns:
        tokenized (list): List of tokenized text, each element either a string or tuple.
    """

    tokens = nlp(text)
    assert len(tokens) > 0, 'Zero tokens were found in text: %s' % text
    assert text == ''.join(t.text_with_ws for t in tokens), 'Detokenization failed: %s' % text

    phrase = ''
    tokenized = []
    for in_pos_filter, group in groupby(tokens, lambda t: t.pos_ == pos_filter or '\n' in t.text):
        phrase = ''.join([t.text_with_ws for t in group])
        if in_pos_filter or not phrase.split():
            for is_punct, chars in groupby(phrase, lambda t: t in string.punctuation + ' \n'):
                phrase = ''.join([c for c in chars])
                if is_punct:
                    tokenized.append(phrase)
                else:
                    tokenized.append(('phrase_to_phonemize', phrase))
                    if phrase[-1:] == ' ':
                        tokenized.append(' ')
        else:
            tokenized.append(('phrase_to_phonemize', phrase))
            if phrase[-1:] == ' ':
                tokenized.append(' ')

    return tokenized


def _phonemize_phrases(phrases, language='en-us', **kwargs):
    """
    Convert graphemes of a phrase to phonemes.

    Args:
        phrase (str)
        language (str): Translation language rules to apply to `phonemizer`, e.g. 'en-us'
        **kwargs: Utility arguments passed to `phonemizer.phonemize`

    Returns:
        ph (str) Phonemic representation of phrase
    """

    ph = phonemize(
        phrases,
        **kwargs,
        separator=_separator,
        strip=True,
        with_stress=True,
        njobs=os.cpu_count(),
        backend='espeak',
        language=language)

    return ph


def filter_(function, dataset):
    """ Similar to the python ``filter`` function with extra logging.

    Args:
        function (callable)
        dataset (iterable of TextSpeechRow)

    Returns:
        iterable of TextSpeechRow
    """
    bools = [function(e) for e in dataset]
    positive = [e for i, e in enumerate(dataset) if bools[i]]
    negative = [e for i, e in enumerate(dataset) if not bools[i]]
    speaker_distribution = Counter([example.speaker for example in negative])
    if len(negative) != 0:
        logger.info('Filtered out %d examples via ``%s`` with a speaker distribution of:\n%s',
                    len(negative), function.__name__, pprint.pformat(speaker_distribution))
    return positive


def _dataset_loader(
    root_directory_name,
    url,
    speaker,
    url_filename=None,
    create_root=False,
    check_files=['{metadata_filename}'],
    directory=DATA_PATH,
    metadata_filename='{directory}/{root_directory_name}/metadata.csv',
    metadata_text_column='Content',
    metadata_audio_column='WAV Filename',
    metadata_audio_path='{directory}/{root_directory_name}/wavs/{metadata_audio_column_value}',
    **kwargs,
):
    """ Load a standard speech dataset.

    A standard speech dataset has these invariants:
        - The file structure is similar to:
            {root_directory_name}/
                metadata.csv
                wavs/
                    audio1.wav
                    audio2.wav
        - The metadata CSV file contains a mapping of audio transcriptions to audio filenames.
        - The dataset contains one speaker.
        - The dataset is stored in a ``tar`` or ``zip`` at some url.

    Args:
        root_directory_name (str): Name of the directory inside `directory` to store data. With
            `create_root=False`, this assumes the directory will be created while extracting
            `url`.
        url (str): URL of the dataset file.
        speaker (src.datasets.Speaker): The dataset speaker.
        create_root (bool, optional): If ``True`` extract tar into
            ``{directory}/{root_directory_name}``. The file is downloaded into
            ``{directory}/{root_directory_name}``.
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
    logger.info('Loading `%s` speech dataset', root_directory_name)
    directory = Path(directory)
    metadata_filename = metadata_filename.format(
        directory=directory, root_directory_name=root_directory_name)
    check_files = [
        str(Path(f.format(metadata_filename=metadata_filename)).absolute()) for f in check_files
    ]

    if create_root:
        (directory / root_directory_name).mkdir(exist_ok=True)

    download_file_maybe_extract(
        url=url,
        directory=str((directory / root_directory_name if create_root else directory).absolute()),
        check_files=check_files,
        filename=url_filename)
    dataframe = pandas.read_csv(Path(metadata_filename), **kwargs)
    return [
        TextSpeechRow(
            text=row[metadata_text_column].strip(),
            audio_path=Path(
                metadata_audio_path.format(
                    directory=directory,
                    root_directory_name=root_directory_name,
                    metadata_audio_column_value=row[metadata_audio_column])),
            speaker=speaker,
            metadata={
                k: v
                for k, v in row.items()
                if k not in [metadata_text_column, metadata_audio_column]
            })
        for _, row in dataframe.iterrows()
    ]

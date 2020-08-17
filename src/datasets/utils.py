from collections import Counter
from functools import partial
from pathlib import Path

import logging
import os
import pprint

from hparams import configurable
from hparams import HParam
from multiprocessing.pool import ThreadPool
from third_party import LazyLoader
from torchnlp.download import download_file_maybe_extract
from tqdm import tqdm

import torch
librosa = LazyLoader('librosa', globals(), 'librosa')
pandas = LazyLoader('pandas', globals(), 'pandas')

from src.audio import cache_get_audio_metadata
from src.audio import get_signal_to_db_mel_spectrogram
from src.audio import to_floating_point_pcm
from src.audio import normalize_audio
from src.audio import pad_remainder
from src.audio import read_audio
from src.datasets.constants import TextSpeechRow
from src.environment import DATA_PATH
from src.environment import IS_TESTING_ENVIRONMENT
from src.environment import ROOT_PATH
from src.environment import TEMP_PATH
from src.environment import TTS_DISK_CACHE_NAME
from src.utils import batch_predict_spectrograms
from src.utils import OnDiskTensor

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
        _, trim = librosa.effects.trim(to_floating_point_pcm(signal))
        signal = signal[trim[0]:trim[1]]

        # TODO: Now that `get_signal_to_db_mel_spectrogram` is implemented in PyTorch, we could
        # batch process spectrograms. This would likely be faster. Also, it could be fast to
        # compute spectrograms on-demand.
        with torch.no_grad():
            db_mel_spectrogram = get_signal_to_db_mel_spectrogram()(
                torch.tensor(to_floating_point_pcm(signal), requires_grad=False), aligned=True)
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
    with ThreadPool(1 if IS_TESTING_ENVIRONMENT else os.cpu_count()) as pool:
        # NOTE: `chunksize` with `imap` is more performant while allowing us to measure progress.
        # TODO: Consider using `imap_unordered` instead of `imap` because it is more performant,
        # learn more:
        # https://stackoverflow.com/questions/19063238/in-what-situation-do-we-need-to-use-multiprocessing-pool-imap-unordered
        # However, it's important to return the results in the same order as they came.
        return list(tqdm(pool.imap(partial_, data, chunksize=128), total=len(data)))


def _normalize_audio_column_helper(example):
    return example._replace(audio_path=normalize_audio(example.audio_path))


def normalize_audio_column(data):
    """ For each example in ``data``, normalize the audio using SoX and update the ``audio_path``.

    Args:
        data (iterables of TextSpeechRow)

    Returns:
        (iterable of TextSpeechRow): Iterable of text speech rows with ``audio_path`` updated.
    """
    cache_get_audio_metadata([e.audio_path for e in data])

    logger.info('Normalizing dataset audio using SoX.')
    with ThreadPool(1 if IS_TESTING_ENVIRONMENT else os.cpu_count()) as pool:
        # NOTE: `chunksize` allows `imap` to be much more performant while allowing us to measure
        # progress.
        iterator = pool.imap(_normalize_audio_column_helper, data, chunksize=1024)
        return_ = list(tqdm(iterator, total=len(data)))

    # NOTE: `cache_get_audio_metadata` for any new normalized audio paths.
    cache_get_audio_metadata([e.audio_path for e in return_])

    return return_


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


def alignment_dataset_loader(alignments='gs://wellsaid_labs_datasets/hilary_noriega/alignments'):
    """ Load an alignment text to speech (TTS) dataset.

    IDEAS:
    1. Preprocess the audio spectrogram before hand, and slice from it. We'd still need to adjust
       the edges. This might not be the biggest issue, since a typical spectrogram is 100ms.
       This might cause issues downstream for the signal model.
    2. Measure attention skipping while training
    3. Measure the difference between the first half and last half of the audio for loudness.

    QUESTIONS:
    1. Predicted spectrogram... How long does it take to compute a predicted spectrogram? We
    could just have a sampler that samples the first 100,000 rows, and computes for that. That's
    assuming that doing so ondemand is too slow. In order to do so ondemand, we'd want to compute
    128 aligned examples every half a second? The other option... is that we could increasingly
    grow the size of the dataset (with a cap) as training is going on. We could always have a
    process in the background that's processing more predicted spectrograms, and meanwhile the
    training continues to go.

    CONCERNS:
    1. Generators cannot be pickled; thefore, we'll need to recreate it. Or we'll need to create
       the generator directly in the child process.
    2. We cannot compute all the spectrograms, and we might not be able to compute the spectrograms
       fast enough to-do so on-demand. We'll want to develop a sampler that can backfill. As
       training goes on, it'll progressively add more data to the dataset.
       The sampler can just get the next sample, if it's ready. If it's not ready, then it can
       grab a previous example from the last 1,000 or so examples.
       The reason I like this is it simplifies the pipeline. We don't need batch processing before
       hand in order to ensure performance. That'll simplify the scripting process, wohoo!
       The reason I don't like this is it adds randomness that cannot be controled. I think
       the randomness is "Okay". We'll be able to replace that component with a determinisitc one
       by pregenerating a large dataset, and just using it.
    3. We'll need some sort of "Universal" encoder and decoder:
       (This should be doable but it'll require some thought. We'll need to worry about the data)
       1. We could have a vocab encoder which is automatically generated based types:
          1. Number (Identity) [1.5, 2.5, 3.5, 4.5]
          2. torch.Tensor (Identity)
          3. String (Label / Tokenizer)
    4. How do we handle the discrepency between annotated data, and non-annotated data?
    5. Most likely, our users will be using the system without annotations. It should be capable
       of operating without annotations required.
    6. How do we break up the program?
       1. Loading
       2. Variety of modules for annotating
       3. Some universal example, that can be passed downstream.

    7. What does realistic usage look like? Should we annotate everything, and then use dropout?
       Should we randomly annotate?
       Always having annotations is appealing because it'll make modeling easier but that doesn't
       make sense in real life. In real-life, we'll sometimes have annotations and sometimes we
       won't.
       The idea of dropout is interesting... We shouldn't use dropout. IT won't be as effective.
       We should dropout entire annotations, similar to "block dropout" which is frequently more
       effective. A user is likely including or not an annotation...
    8. What does deployment look like? Which annotations will we add?
       We'll automatically parse with spacy, and lower case the text. The loudness / speed / pitch
       will not be defined.
       We might take advantage of the metadata or preset it.
    9. Should the server parse the XML? Or should that be responsiblity of the frontend?
       This should be handled on the backend, and that'll make the API easier to use.

    TODO:
    1. Cache alignments, voice-over, and voice-over script.
    2. Preprocess voice-over with `SoX` or `ffmpeg` and cache.
    3. Pick the script subset to generate data from.
    4. Log the number of hours of data
    4. Start an infinite loop to generate slices:
      1. Randomly select a starting alignment.
      2. Randomly select an ending alignment that is less that the maximum seconds.
      3. Reject the slice if:
          1. There is an unalignment in the middle of the slice.
          2. The start or end alignment has too little audio per character.
          3. There is a number in the slice.
          3. Research others...
      4. Normalize script text.
      5. Preprocess the script with spaCy medium, and add both `.tensor` and `.vector` features to
         the slice.
      6. For a random number of non-overlapping random slices, compute the:
          - Average and rounded (to prevent overfitting) loudness (with ITU-R BS.1770-4).
          - Average and rounded (to prevent overfitting) speed (seconds per phoneme).
          - Average and rounded (to prevent overfitting) pitch (with CREPE or SPICE or torchaudio).
      7. For a random number of transitions, compute the (rounded to prevent overfitting) pause
         time in seconds.
      8. Lower case the text, and provide an extra feature with regard to capitalization.
      9. For a random number of words, provide the phonetic spelling.
      10. For the audio file, compute the noise by measuring the quietest second. Provide that
          information. (Later)
      11. Extract and provide any related metadata with regards to book or article.
    5. Following getting the dataset... the spectrogram is generated.
    6. The spectrogram shapes are cached. In order to do so, we'll just need to pregenerate
       100 * batch_size, and the examples are sorted internally.
    7. The grapheme to phoneme data is cached. This is not required iff we have a on-demand workflow
       with backfilling.
    8. The iterators are passed onto the processes. They need to be recreated.
    We can then chain a sampler onto this generator for data sampling...
    """
    pass


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
        url_filename (str, optional): Name of the file downloaded; Otherwise, a filename is
            extracted from the url.
        create_root (bool, optional): If ``True`` extract tar into
            ``{directory}/{root_directory_name}``. The file is downloaded into
            ``{directory}/{root_directory_name}``.
        check_files (list of str, optional): The download is considered successful, if these files
            exist.
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

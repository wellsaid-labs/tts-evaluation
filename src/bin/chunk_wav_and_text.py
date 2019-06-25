""" Script for chunking a transcript with the associated audio file.

Prior:

    Setup a service account for your local machine that allows you to write to the
    `wellsaid-speech-to-text` bucket in Google Cloud Storage. Start by creating a service account
    using the Google Cloud Console. Give the service account the
    "Google Cloud Storage Object Admin" role (f.y.i. the "Role" menu scrolls). Name the service
    account something similar to "michael-local-gcs". Download a key as JSON and put it somewhere
    secure locally:

    $ mv ~/Downloads/WellSaidLabs-bd1l2j3opsc.json env/gcs/credentials.json

NOTE: `*[!)].wav` This pattern excludes any wav file with a ")" before the ".wav". We use this to
  exclude normalized audio files.

Example:

    GOOGLE_APPLICATION_CREDENTIALS=env/gcs/credentials.json \
    python3 -m src.bin.chunk_wav_and_text --wav 'data/other/Heather/wavs/*[!)].wav' \
                                          --csv 'data/other/Heather/csvs/*.csv' \
                                          --destination data/other/Heather/dest/

Batch Example:

    for directory in 'data/Beth' 'data/Heather' 'data/Hilary' 'data/Sam' 'data/Susan'
    do
        GOOGLE_APPLICATION_CREDENTIALS=env/gcs/credentials.json \
        python3 -m src.bin.chunk_wav_and_text --wav "$directory/03 Recordings/*[!)].wav" \
                                              --csv "$directory/04 Scripts (CSV)/*.csv" \
                                              --destination "$directory/05 Processed/"
    done

Compression:

    Use ``tar -czvf name-of-archive.tar.gz /path/to/directory-or-file`` to compress the archive. For
    those using Mac OS do not use "compress" to create a `.zip` file instead [1].

[1] Mac OS uses Archive Utility to compress a directory creaing by default a
"Compressed UNIX CPIO Archive file" (CPGZ) under the `.zip` extension. The CPGZ is created with
Apple's "Apple gzip" that a Linux gzip implementations are unable to handle.
https://www.intego.com/mac-security-blog/understanding-compressed-files-and-apples-archive-utility/
"""
from collections import namedtuple
from itertools import groupby
from multiprocessing.pool import ThreadPool
from pathlib import Path
from tqdm import tqdm

import argparse
import json
import logging
import re
import requests
import statistics
import subprocess
import time
import unidecode

import librosa
import pandas

from google.cloud import speech
from google.cloud.speech import types
from google.protobuf.json_format import MessageToDict
from Levenshtein import distance as get_edit_distance

from src.audio import normalize_audio
from src.audio import read_audio
from src.hparams import configurable
from src.hparams import ConfiguredArg
from src.hparams import set_hparams
from src.utils import align_tokens
from src.utils import flatten
from src.utils import record_stream

TERMINAL_COLOR_RESET = '\033[0m'
TERMINAL_COLOR_RED = '\033[91m'
TERMINAL_COLOR_BLUE = '\033[94m'
TERMINAL_COLOR_PURPLE = '\033[95m'

logging.basicConfig(
    format='{}%(levelname)s:%(name)s:{} %(message)s'.format(TERMINAL_COLOR_PURPLE,
                                                            TERMINAL_COLOR_RESET),
    level=logging.INFO)
logger = logging.getLogger(__name__)

# An alignment or the absence of an alignment between the text and audio.
#
# Args:
#     start_audio (int): Start of the audio in samples.
#     end_audio (int): End of audio in samples.
#     start_text (int): Start of text in characters.
#     end_text (int): End of text in characters.
Alignment = namedtuple('Alignment', ['start_text', 'end_text', 'start_audio', 'end_audio'])
Nonalignment = namedtuple('Nonalignment', ['start_text', 'end_text'])
Nonalignment.__new__.__defaults__ = (None, None)
Alignment.__new__.__defaults__ = (None, None)


def natural_keys(text):
    """ Returns keys (`list`) for sorting in a "natural" order.

    Inspired by: http://nedbatchelder.com/blog/200712/human_sorting.html
    """
    return [(int(char) if char.isdigit() else char) for char in re.split(r'(\d+)', str(text))]


@configurable
def seconds_to_samples(seconds, sample_rate=ConfiguredArg()):
    """
    Args:
        seconds (str or float): Number of seconds.

    Returns:
        (int): Number of samples.

    Example:
        >>> seconds_to_samples('0.1s', 24000)
        2400
        >>> seconds_to_samples(0.1, 24000)
        2400
    """
    if isinstance(seconds, str) and seconds[-1] == 's':
        seconds = float(seconds[:-1])

    return int(round(seconds * sample_rate))


@configurable
def samples_to_seconds(samples, sample_rate=ConfiguredArg()):
    """
    Args:
        samples (int): Number of samples.

    Returns:
        (float): Number of seconds.

    Example:
        >>> samples_to_seconds(2400, 24000)
        0.1
    """
    return samples / sample_rate


def request_google_sst(wav_paths,
                       sst_cache_directory,
                       destination_name,
                       gcs_bucket_name='wellsaid-speech-to-text',
                       sst_use_enhanced=True,
                       sst_enable_automatic_punctuation=True,
                       sst_model='video'):
    """ Request asynchronous google speech-to-text on `wav_path`.

    The `sst_use_enhanced` and `sst_model` defaults are set based on:
    https://blog.timbunce.org/2018/05/15/a-comparison-of-automatic-speech-recognition-asr-systems/

    Args:
        wav_paths (list of Path): The audio file path.
        sst_cache_directory (Path): Directory used to cache speech-to-text requests locally.
        destination_name (str): Directory used to denominate this execution in GCS.
        gcs_bucket_name (string, optional): An existing GCS bucket to store files for SST.
        sst_use_enhanced (bool, optional): Refer to the docs for `RecognitionConfig` to learn more.
        sst_enable_automatic_punctuation (bool, optional): Refer to the docs for `RecognitionConfig`
            to learn more.
        sst_model (str, optional): Refer to the docs for `RecognitionConfig` to learn more.

    Returns:
        (list) [
          {
            transcript (str): Transcript of the audio chunk.
            confidence (float): Google's confidence in the transcript.
            words (list): List of words in the transcript tokenized by spaces.
            [
              {
                startTime (str): Start time of the word in seconds. (i.e. "0s")
                endTime (str): End time of the word in seconds. (i.e. "0.100s")
                word (str)
              },
              {
                ...
              }
            ]
          },
          {
            ...
          }
        ]
    """
    sst_caches = [(sst_cache_directory / p.name).with_suffix('.json') for p in wav_paths]
    if all(p.exists() for p in sst_caches):
        results = [json.loads(p.read_text())['results'] for p in sst_caches]
        return [[t['alternatives'][0] for t in r if t['alternatives'][0]] for r in results]

    # Check via `gsutil -q stat` if the upload already occured.
    gcs_base_uri = 'gs://%s/%s' % (gcs_bucket_name, destination_name)
    logger.info('Uploading files to GCS at %s', gcs_base_uri)
    # NOTE: There is no functionality to expose a progress bar in the Google Python library.
    # https://github.com/googleapis/google-cloud-python/issues/1077
    # Upload with `gsutil` for the progress bar.
    # `-m`: Performs the operation using a combination of multi-threading and multi-processing.
    # `-n`: When specified, existing files or objects at the destination will not be overwritten.
    subprocess.run(['gsutil', '-m', 'cp', '-n'] + wav_paths + [gcs_base_uri], check=True)

    def _helper(index, wav_path, sst_cache):
        gcs_uri = '%s/%s' % (gcs_base_uri, wav_path.name)
        client = speech.SpeechClient()
        audio = types.RecognitionAudio(uri=gcs_uri)
        # Note that the documentation for `RecognitionConfig` does not include all the options.
        # All the potential parameters are included here:
        # https://cloud.google.com/speech-to-text/docs/reference/rest/v1p1beta1/RecognitionConfig
        config = types.RecognitionConfig(
            language_code='en-US',
            model=sst_model,
            use_enhanced=sst_use_enhanced,
            enable_automatic_punctuation=sst_enable_automatic_punctuation,
            enable_word_time_offsets=True)
        # NOTE: Despite that this API is asynchronous, it cannot be used to launch multiple
        # parallel operations.
        operation = client.long_running_recognize(config, audio)
        with tqdm(total=100, position=index, desc=wav_path.stem) as progress_bar:
            while not operation.done():
                if operation.metadata is not None:
                    metadata = MessageToDict(operation.metadata)
                    if 'progressPercent' in metadata:
                        progress_bar.update(metadata['progressPercent'] - progress_bar.n)
                # NOTE: Careful not to exceed default 300 requests a second quota; however, if you
                # end up exceeding the quota, Google will force you to wait up to 30 minutes before
                # it allows more requests.
                time.sleep(60)
            progress_bar.update(100 - progress_bar.n)

        sst_response = operation.result()
        sst_response = MessageToDict(sst_response)
        sst_cache.write_text(json.dumps(sst_response))
        return sst_response['results']

    pool = ThreadPool(len(wav_paths))
    sst_responses = pool.starmap(_helper, list(zip(range(len(wav_paths)), wav_paths, sst_caches)))
    pool.close()
    pool.join()

    # NOTE: SST provides only one alternative unless configured otherwise.
    return [[t['alternatives'][0] for t in r if t['alternatives'][0]] for r in sst_responses]


def allow_substitution(a, b, max_length_difference=2):
    """ Given similar context, determine if token `a` and `b` are similar enough to align.

    Args:
        a (str): Token a.
        b (str): Token b.
        max_length_difference (int): The maximum difference in length allowed.

    Returns:
        (bool)
    """
    if abs(len(a) - len(b)) >= max_length_difference:
        return False

    if get_edit_distance(remove_punctuation(a.lower()), remove_punctuation(
            b.lower())) > max(len(a), len(b)) / 2:
        return False

    return True


def remove_punctuation(string):
    """ Remove all punctuation from a string.

    Example:
        >>> remove_punctuation('123 abc !.?')
        '123 abc '
    """
    return re.sub(r'[^\w\s]', '', string)


def log_comparison_of_sst_to_transcript_tokens(transcript_tokens, sst_tokens):
    """ Print helpful statistics comparing transcript tokens and SST tokens.

    Args:
        sst_tokens (list of dict)
        transcript_tokens (list of dict)
    """
    logger.info('Token count for the SST transcript: %d', len(sst_tokens))
    logger.info('Token count for the original transcript: %d', len(transcript_tokens))

    # Display character discrepancies between the transcripts to highlight puncutation differences.
    format_characters = lambda l: ''.join(sorted(list(set(l))))
    sst_characters = format_characters(flatten([t['text'] for t in sst_tokens]))
    transcript_characters = format_characters(flatten([t['text'] for t in transcript_tokens]))
    logger.info('The SST transcript includes these characters: %s', sst_characters)
    logger.info('The original transcript includes these characters: %s', transcript_characters)


@configurable
def align_wav_and_scripts(sst_results, scripts, sample_rate=ConfiguredArg()):
    """ Align an audio file with the scripts spoken in the audio.

    NOTE: SST stands for Speech-to-Text.
    NOTE: There are a number of alignment issues due to dashes. There is no "simple" method for
    resolving this issue because during alignment either the transcript includes dashes or Google's
    punctuationless SST does.
    NOTE: This approach for alignment is accurate but relies on white space tokenization. With white
    space tokenization, it's easy to reconstruct text spans; however, this approach messes up in
    cases like "parents...and", "tax-deferred", "little-c/Big-C" and "1978.In".

    TODO:
        - Make the `scripts` parameter optional incase a transcript is not available.
        - Sort alignments by the ratio of text length to audio length because tokens with a smaller
          ratio than usual have a potentially problematic alignment.
        - Consider an approach where:
          1. Align the SST transcript to the original transcript without sentence-level punctuation
             avoiding the error cases mentioned above.
          2. Align pervious alignment with the white space tokenized original transcript tokens.
          This approach should succeed to align "1978.In" and "parents...and"; however, it would
          still fail on "tax-deferred" because Google's SST will still include dashes in their
          output.

    Args:
        sst_results (list): Speech-to-text alignment.
        scripts (list of str): The scripts spoken in the audio.
        sample_rate (int): The sample rate of the audio.

    Returns:
        (list of Alignment or Nonalignment): List of scripts and their corresponding token
            alignments.
    """
    # Tokenize tokens similar to Google SST for alignment.
    # NOTE: Google SST as of June 21st, 2019 tends to tokenize based on white spaces.
    script_tokens = flatten([[{
        'text': m.group(0),
        'start_text': m.start(),
        'end_text': m.end(),
        'script_index': i
    } for m in re.finditer(r'\S+', script)] for i, script in enumerate(scripts)])
    sst_tokens = flatten([[{
        'text': w['word'],
        'start_audio': seconds_to_samples(w['startTime'], sample_rate),
        'end_audio': seconds_to_samples(w['endTime'], sample_rate),
    } for w in r['words']] for r in sst_results])
    log_comparison_of_sst_to_transcript_tokens(script_tokens, sst_tokens)

    logger.info('Aligning transcript with SST.')
    # `.lower()` to assist in alignment in cases like "HEALTH" and "health".
    # NOTE: The 10% window has performed well based off empirical evidence while a 5% window
    # did not; furthermore, a 10% window is justified assuming a 7 - 8% insertion and deletion
    # rate following an alignment.
    # NOTE: The `max(x, 10)` is present for testing small sequences.
    alignment_window = max(round(max(len(sst_tokens), len(script_tokens)) * 0.1), 10)
    num_unaligned, alignments = align_tokens([t['text'].lower() for t in script_tokens],
                                             [t['text'].lower() for t in sst_tokens],
                                             alignment_window,
                                             allow_substitution=allow_substitution)

    # Create list of `Alignment` and `Nonalignment`
    num_unaligned_tokens = len(script_tokens) + len(sst_tokens) - len(alignments) * 2
    alignment = alignments.pop(0)
    scripts_alignments = [[] for _ in range(len(scripts))]
    irregular_alignements = []
    for i, script_token in enumerate(script_tokens):
        text_span = (script_token['start_text'], script_token['end_text'])
        script_alignments = scripts_alignments[script_token['script_index']]

        if alignment is None or i != alignment[0]:
            script_alignments.append(Nonalignment(*text_span))
        else:
            sst_token = sst_tokens[alignment[1]]
            audio_span = (sst_token['start_audio'], sst_token['end_audio'])
            script_alignments.append(Alignment(*text_span, *audio_span))
            alignment = alignments.pop(0) if len(alignments) > 0 else None

            # Log statistics on tokens that do not match.
            sst_token_text = sst_token['text'].lower()
            script_token_text = script_token['text'].lower()
            if sst_token_text != script_token_text:
                distance = get_edit_distance(script_token_text, sst_token_text)
                irregular_alignements.append((script_token_text, sst_token_text, distance))
            if remove_punctuation(sst_token_text) != remove_punctuation(script_token_text):
                num_unaligned_tokens += 1

    irregular_alignements = sorted(irregular_alignements, key=lambda a: a[2], reverse=True)
    logger.warning('%d script tokens and SST tokens do not match, these are the top hundred: %s',
                   len(irregular_alignements), irregular_alignements[:100])
    logger.info('Word error rate of SST (no punctuation or capitalization): %f%% [%d of %d]',
                num_unaligned_tokens / len(script_tokens) * 100, num_unaligned_tokens,
                len(script_tokens))

    # Print consecutive unaligned spans.
    unaligned_substrings = []
    for script, script_alignments in zip(scripts, scripts_alignments):
        spans = [list(s) for a, s in groupby(script_alignments) if type(a) == Nonalignment]
        if len(spans) > 0:
            unaligned_substrings += [script[s[0].start_text:s[-1].end_text] for s in spans]
    unaligned_substrings = sorted(unaligned_substrings, key=len, reverse=True)
    logger.warning('Unaligned text spans between SST and transcript: %s', unaligned_substrings)

    num_not_aligned = sum([type(a) == Nonalignment for a in flatten(scripts_alignments)])
    logger.info('Failed to align %f%% [%d of %d] of transcript tokens.',
                num_not_aligned / len(script_tokens) * 100, num_not_aligned, len(script_tokens))

    return scripts_alignments


def review_chunk_alignments(script, spans):
    """ Check some invariants and log warnings for ``chunk_alignments``

    NOTE:
        - Refer to ``chunk_alignments`` to better understand arguments.

    Args:
        script (str): Script to review.
        spans (list of dict): Spans in sorted order.

    Returns:
        unaligned_substrings (list of str): List of substrings that were not aligned in a span.
    """
    # Ensure that the spans are in sorted order and do not overlap.
    for i in range(len(spans) - 1):
        if not spans[i]['text'][1] <= spans[i + 1]['text'][0]:
            raise ValueError('Text must be monotonically increasing.')
        if not spans[i]['audio'][1] <= spans[i + 1]['audio'][0]:
            raise ValueError('Audio must be monotonically increasing.')

    unaligned_substrings = []
    last_span_end_index = 0
    for span in spans:
        if last_span_end_index != span['text'][0]:
            unaligned_substrings.append(script[last_span_end_index:span['text'][0]].strip())
        last_span_end_index = span['text'][1]
    if last_span_end_index != len(script):
        unaligned_substrings.append(script[last_span_end_index:len(script)].strip())

    unaligned_substrings = [s for s in unaligned_substrings if len(s) > 0]
    num_unaligned_characters = sum([len(s) for s in unaligned_substrings])
    if num_unaligned_characters != 0:
        # NOTE: Insert largest to smallest index.
        to_print = list(script) + [TERMINAL_COLOR_RESET]
        for span in reversed(spans):
            to_print.insert(
                span['text'][1],
                '%s»%s%s' % (TERMINAL_COLOR_BLUE, TERMINAL_COLOR_RESET, TERMINAL_COLOR_RED))
            to_print.insert(
                span['text'][0],
                '%s%s«%s' % (TERMINAL_COLOR_RESET, TERMINAL_COLOR_BLUE, TERMINAL_COLOR_RESET))
        to_print = [TERMINAL_COLOR_RED] + to_print

        # NOTE: Errors will only appear on the beginning or end of the script. The chunking
        # algorithm does not cut on words that won't align.
        logger.warning('Unable to align %f%% (%d of %d) of script, see here - \n%s',
                       num_unaligned_characters / len(script) * 100, num_unaligned_characters,
                       len(script), ''.join(to_print))

    return unaligned_substrings


def average_silence_delimiter(alignment, next_alignment):
    """ Compute the average samples of silence per character

    Args:
        alignment (Alignment or Nonalignment)
        next_alignment (Alignment or Nonalignment): Alignment following `alignment`.

    Returns:
        (float): Average silence per character between two alignments or -1.0 if either alignment
          is a `Nonalignment`.
    """
    if next_alignment.start_text <= alignment.end_text:
        raise ValueError('Aligment and next alignment have to be disjointed and in sequence.')

    if isinstance(alignment, Nonalignment) or isinstance(next_alignment, Nonalignment):
        return -1.0

    return (next_alignment.start_audio - alignment.end_audio) / (
        next_alignment.start_text - alignment.end_text)


@configurable
def chunk_alignments(alignments,
                     script,
                     max_chunk_seconds,
                     sample_rate=ConfiguredArg(),
                     delimiter=average_silence_delimiter):
    """ Chunk audio and text to be less than ``max_chunk_seconds``.

    Notes:
        - Given `delimiter=average_silence_delimiter`, this chunks based on the largest silence
          between alignments. The assumption is that the returned chunks are independent
          contextually because there exists a large silence.
        - Ideally chunk independence means that prior text does not affect the voice actors
          interpretation of that chunk's text.

    TODO:
        - Consider instead of chunking up based on a delimiter, creating random cuts.
        - Consider adding noise to our cuts instead of always cutting on the longest silence.
        - Write a more efficient algorithm that doesn't recompute the silences and
          chunks every iteration.
        - Consider deliminating on POS phrases instead of silence; therefore, not biasing
          the model towards smaller silences.

    Args:
        alignments (list of Alignment): List of alignments between text and audio.
        script (str): Transcript alignments refer too.
        max_chunk_samples (int): Number of samples for the maximum chunk audio length.
        sample_rate (int): The sample rate of the audio.
        delimiter (callable, optional): Given an alignment returns -1 or a positive floating point
            number. A larger number is more likely to be used to split a sequence.

    Returns:
        spans (list of dict {
          text (tuple(int, int)): Span of text as measured in characters.
          audio (tuple(int, int)): Span of audio as measured in samples.
        }
        (list of str): List of substrings that were not aligned in a span.
    """
    max_chunk_samples = seconds_to_samples(max_chunk_seconds, sample_rate=sample_rate)
    chunks = []
    # NOTE: We cannot end / start on a Nonalignment; therefore, we cut them off.
    max_chunk = alignments
    while len(max_chunk) > 0 and isinstance(max_chunk[0], Nonalignment):
        max_chunk = max_chunk[1:]
    while len(max_chunk) > 0 and isinstance(max_chunk[-1], Nonalignment):
        max_chunk = max_chunk[:-1]
    if len(max_chunk) == 0:
        return [], (len(script), len(script))

    # Chunk the max chunk until the max chunk is smaller than ``max_chunk_samples``
    get_chunk_length = lambda chunk: chunk[-1].end_audio - chunk[0].start_audio
    while (max_chunk is not None and len(max_chunk) > 1 and
           get_chunk_length(max_chunk) > max_chunk_samples):
        silences = [delimiter(a, max_chunk[i + 1]) for i, a in enumerate(max_chunk[:-1])]
        max_silence = max(silences)
        max_silences = [i for i, s in enumerate(silences) if s == max_silence]
        # Pick the max silence closest to the middle of the sequence given multiple large silences.
        max_silence_index = min(max_silences, key=lambda i: abs(i - round(len(silences) / 2)))

        if max_silence != -1:
            chunks.append(max_chunk[:max_silence_index + 1])
            chunks.append(max_chunk[max_silence_index + 1:])
        else:  # NOTE: If ``delimiter`` suggests no cuts, then we ignore ``max_chunk```
            max_chunk_text = script[max_chunk[0].start_text:max_chunk[-1].end_text]
            logger.warning('Unable to cut:\n%s%s%s', TERMINAL_COLOR_RED, max_chunk_text,
                           TERMINAL_COLOR_RESET)

        max_chunk = None  # NOTE: ``max_chunk``` has been processed.

        if len(chunks) > 0:
            max_chunk_index, _ = max(enumerate(chunks), key=lambda a: get_chunk_length(a[1]))
            max_chunk = chunks.pop(max_chunk_index)

    if max_chunk is not None:
        chunks.append(max_chunk)

    spans = [{
        'text': (chunk[0].start_text, chunk[-1].end_text),
        'audio': (chunk[0].start_audio, chunk[-1].end_audio),
    } for chunk in chunks]
    spans = sorted(spans, key=lambda s: s['text'][0])
    return spans, review_chunk_alignments(script, spans)


def setup_io(wav_pattern,
             csv_pattern,
             destination,
             wav_directory_name='wavs',
             csv_metadata_name='metadata.csv',
             sst_cache_name='.sst'):
    """ Perform basic IO operations to setup this script.

    The WAV paths and CSV paths are matched up based on a natural sorting algorithm: `natural_keys`.

    Args:
        wav_pattern (str): The audio file glob pattern.
        csv_pattern (str): The CSV file globl pattern containing metadata associated with audio
            file.
        destination (Path): Directory to save the processed data.
        wav_directory_name (str, optional): The directory name to store audio clips.
        csv_metadata_name (str, optional): The filename for metadata.
        sst_cache_name (str, optional): The directory name for SST files.

    Returns:
        wav_paths (list of Path): WAV files to process.
        csv_paths (list of Path): CSV files to process.
        wav_directory (Path): The directory in which to store audio clips.
        sst_cache_directory (Path): The directory in which to cache SST requests.
        metadata_filename (Path): The filename to write CSV metadata to.
    """
    metadata_filename = destination / csv_metadata_name  # Filename to store CSV metadata
    wav_directory = destination / wav_directory_name  # Directory to store clips
    sst_cache_directory = destination / sst_cache_name
    wav_directory.mkdir(parents=True, exist_ok=True)
    sst_cache_directory.mkdir(exist_ok=True)

    record_stream(destination)

    wav_paths = sorted(list(Path('.').glob(wav_pattern)), key=natural_keys)
    csv_paths = sorted(list(Path('.').glob(csv_pattern)), key=natural_keys)

    # Check invariants
    assert all(
        [wav_path.suffix == '.wav' for wav_path in wav_paths]), 'Please select only WAV files'
    assert all(
        [csv_path.suffix == '.csv' for csv_path in csv_paths]), 'Please select only CSV files'

    if len(csv_paths) == 0 and len(wav_paths) == 0:
        logger.warning('Found no CSV or WAV files.')
        return
    elif len(csv_paths) != len(wav_paths):
        logger.warning('CSV (%d) and WAV (%d) files are not aligned.', len(csv_paths),
                       len(wav_paths))
        return
    else:
        logger.info('Processing %d files', len(csv_paths))

    logger.info('Found %d CSV and %d WAV files.', len(csv_paths), len(wav_paths))
    return wav_paths, csv_paths, wav_directory, sst_cache_directory, metadata_filename


def normalize_text(text):
    """ Normalize the text such that the text matches up closely to the audio.

    NOTE: These rules are specific to the datasets processed so far.

    Args:
        text (str)

    Returns
        text (str)
    """
    text = text.strip()
    text = text.replace('\t', '  ')
    text = text.replace('®', '')
    # Remove HTML tags
    text = re.sub('<.*?>', '', text)
    # Fix for a missing space between end and beginning of a sentence.
    # Example match deliminated by <>:
    #   the cold w<ar.T>he term 'business ethics'
    text = re.sub(r"([a-z]{2}[.!?])([A-Z])", r"\1 \2", text)
    return unidecode.unidecode(text)


def main(wav_pattern,
         csv_pattern,
         destination,
         text_column='Content',
         wav_column='WAV Filename',
         max_chunk_seconds=16):
    """ Align audio with scripts, then create and save chunks of audio and text.

    TODO: Consider increasing the `max_chunk_seconds` up to 20 seconds in line with the
      M-AILABS dataset; however, the LJ-Speech dataset has a max length of 10 seconds. The max
      chunk seconds should be the maximum words a speaker will read without needing to pause
      or break up the voice-over.

    Args:
        wav_pattern (str): The audio file glob pattern.
        csv_pattern (str): The CSV file glob pattern containing metadata associated with audio file.
        destination (str): Directory to save the processed data.
        text_column (str, optional): The script column in the CSV file.
        wav_column (str, optional): Column name for the new audio filename column.
        max_chunk_seconds (float, optional): Number of seconds for the maximum chunk audio length.
    """
    set_hparams()

    destination = Path(destination)
    (wav_paths, csv_paths, wav_directory, sst_cache_directory, metadata_filename) = setup_io(
        wav_pattern, csv_pattern, destination)

    logger.info('Normalizing audio files...')
    wav_paths = [normalize_audio(p) for p in tqdm(wav_paths)]
    sst_results = request_google_sst(wav_paths, sst_cache_directory, destination.name)

    all_unaligned_substrings = []
    num_characters = 0
    chunk_lengths = []
    for i, (wav_path, csv_path, sst_result) in enumerate(zip(wav_paths, csv_paths, sst_results)):
        print('-' * 100)
        logger.info('Processing %s:%s', wav_path, csv_path)
        script_wav_directory = wav_directory / wav_path.stem  # Directory to resulting audio chunks
        script_wav_directory.mkdir(exist_ok=True)

        logger.info('Reading audio and csv...')
        audio = read_audio(wav_path)
        data_frame = pandas.read_csv(csv_path)
        if 'Index' in data_frame:  # NOTE: Some CSV files include a unhelpful `Index` column.
            del data_frame['Index']
        scripts = [normalize_text(str(x)) for x in data_frame[text_column]]
        num_characters += sum([len(s) for s in scripts])

        try:
            alignments = align_wav_and_scripts(sst_result, scripts)
        except (requests.exceptions.RequestException, ValueError):
            logger.exception('Failed to align %s with %s', wav_path.name, csv_path.name)
            continue

        logger.info('Chunking and writing...')
        to_write = []
        for j, (script, alignment, row) in enumerate(
                zip(scripts, alignments, data_frame.to_dict('records'))):

            chunks, unaligned_substrings = chunk_alignments(alignment, script, max_chunk_seconds)
            all_unaligned_substrings += unaligned_substrings

            for k, chunk in enumerate(chunks):
                new_row = row.copy()

                new_row[text_column] = script[slice(*chunk['text'])].strip()
                audio_path = script_wav_directory / ('script_%d_chunk_%d.wav' % (j, k))
                new_row[wav_column] = str(audio_path.relative_to(wav_directory))
                to_write.append(new_row)

                audio_span = audio[slice(*chunk['audio'])]
                chunk_lengths.append(samples_to_seconds(chunk['audio'][1] - chunk['audio'][0]))
                librosa.output.write_wav(str(audio_path), audio_span)

        logger.info('Found %d chunks', len(to_write))
        pandas.DataFrame(to_write).to_csv(
            str(metadata_filename), mode='a', header=i == 0, index=False)

    print('=' * 100)
    all_unaligned_substrings = list(map(str, all_unaligned_substrings))
    num_unaligned_characters = sum([len(s) for s in all_unaligned_substrings])
    logger.info('Created %d chunks with mean length %fs ± %f and max length of %fs',
                len(chunk_lengths), statistics.mean(chunk_lengths), statistics.stdev(chunk_lengths),
                max(chunk_lengths))
    logger.warning('Found %f%% [%d of %d] unaligned characters',
                   num_unaligned_characters / num_characters * 100, num_unaligned_characters,
                   num_characters)
    logger.warning('All unaligned substrings:\n%s',
                   sorted(all_unaligned_substrings, key=len, reverse=True))


if __name__ == "__main__":  # pragma: no cover
    # TODO: Consider accepting a list from bash glob.
    parser = argparse.ArgumentParser(description='Align and chunk audio file and text scripts.')
    parser.add_argument('-w', '--wav', type=str, help='Path / Pattern to WAV file to chunk.')
    parser.add_argument('-c', '--csv', type=str, help='Path / Pattern to CSV file with scripts.')
    parser.add_argument('-d', '--destination', type=str, help='Path to save processed files.')
    args = parser.parse_args()
    main(args.wav, args.csv, args.destination)

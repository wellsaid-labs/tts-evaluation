""" Script for generating an alignment between a script and a voice-over.

Given a voice-over of a script, this python script generates an alignment between text spans
and audio spans. Furthermore, this script will help resolve any issues that reduce the accuracy
of the alignment.

Prior:

    Setup a service account for your local machine that allows you to write to the
    `wellsaid_labs_datasets` bucket in Google Cloud Storage. Start by creating a service account
    using the Google Cloud Console. Give the service account the
    "Google Cloud Storage Object Admin" role (f.y.i. the "Role" menu scrolls). Name the service
    account something similar to "michael-gcs-datasets". Download a key as JSON and put it
    somewhere secure locally:

    $ mv ~/Downloads/voice-research-255602-1a5538456fc3.json gcs_credentials.json


Example:

    PREFIX=gs://wellsaid_labs_datasets/hilary_noriega
    GOOGLE_APPLICATION_CREDENTIALS=gcs_credentials.json \
    python -m src.bin.sync_script_with_audio \
      --voice_over "$PREFIX/preprocessed_recordings/Script 1.wav" \
      --voice_over_script "$PREFIX/scripts/Script 1 - Hilary.csv" \
      --destination "$PREFIX/"

Example (Multiple Speakers):

    PREFIX=gs://wellsaid_labs_datasets
    GOOGLE_APPLICATION_CREDENTIALS=gcs_credentials.json \
    python -m src.bin.sync_script_with_audio \
      --voice_over "$PREFIX/hilary_noriega/preprocessed_recordings/Script 1.wav" \
        "$PREFIX/sam_scholl/preprocessed_recordings/WSL Sam4.wav" \
        "$PREFIX/frank_bonacquisti/preprocessed_recordings/WSL-Script 001.wav" \
        "$PREFIX/adrienne_walker_heller/preprocessed_recordings/WSL - Adrienne WalkerScript1.wav" \
        "$PREFIX/heather_doe/preprocessed_recordings/Heather_99-101.wav" \
      --voice_over_script "$PREFIX/hilary_noriega/scripts/Script 1 - Hilary.csv" \
                          "$PREFIX/sam_scholl/scripts/Script 4.csv" \
                          "$PREFIX/frank_bonacquisti/scripts/Script 1.csv" \
                          "$PREFIX/adrienne_walker_heller/scripts/Script 1.csv" \
                          "$PREFIX/heather_doe/scripts/Scripts 99-101.csv" \
      --destination "$PREFIX/hilary_noriega/" \
                    "$PREFIX/sam_scholl/" \
                    "$PREFIX/frank_bonacquisti/" \
                    "$PREFIX/adrienne_walker_heller/" \
                    "$PREFIX/heather_doe/" \
      --sort

Example (Batch):

    PREFIX=gs://wellsaid_labs_datasets/hilary_noriega
    GOOGLE_APPLICATION_CREDENTIALS=gcs_credentials.json \
    python -m src.bin.sync_script_with_audio \
      --voice_over "${(@f)$(gsutil ls "$PREFIX/preprocessed_recordings/*.wav")}" \
      --voice_over_script "${(@f)$(gsutil ls "$PREFIX/scripts/*.csv")}" \
      --destination $PREFIX/ \
      --sort

"""
from collections import namedtuple
from copy import copy
from functools import lru_cache
from io import StringIO
from itertools import chain
from itertools import groupby
from itertools import repeat

import argparse
import json
import logging
import re
import time

import pandas
import unidecode

from google.cloud import speech_v1p1beta1 as speech
from google.cloud import storage
from google.cloud.speech_v1p1beta1 import types
from google.protobuf.json_format import MessageToDict
from tqdm import tqdm

from src.environment import COLORS
from src.environment import set_basic_logging_config
from src.hparams import set_hparams
from src.text import grapheme_to_phoneme
from src.utils import align_tokens
from src.utils import flatten
from src.utils import RecordStandardStreams

set_basic_logging_config()
logger = logging.getLogger(__name__)

# Args:
#   text (str): The script text.
#   start_text (int): The first character offset of the script text.
#   end_text (int): This is equal to: `start_text + len(text)`
#   script_index (int): The index of the script from which the text comes from.
ScriptToken = namedtuple(
    'ScriptToken', ['text', 'start_text', 'end_text', 'script_index'], defaults=(None,) * 4)

# Args:
#   text (str): The script text.
#   start_audio (float): The beginning of the audio span in seconds during which `text` is voiced.
#   end_audio (float): The end of the audio span in seconds during which `text` is voiced.
SttToken = namedtuple('SttToken', ['text', 'start_audio', 'end_audio'], defaults=(None,) * 3)


def format_ratio(a, b):
    """
    Example:
        >>> format_ratio(1, 100)
        '1.000000% [1 of 100]'
    """
    return '%f%% [%d of %d]' % ((float(a) / b) * 100, a, b)


def remove_punctuation(string):
    """ Remove all punctuation from a string.

    Example:
        >>> remove_punctuation('123 abc !.?')
        '123 abc'
    """
    return re.sub(r'[^\w\s]', '', string).strip()


@lru_cache(maxsize=128)
def is_sound_alike(a, b):
    """ Return `True` if `str` `a` and `str` `b` sound a-like. """
    a = unidecode.unidecode(a)
    b = unidecode.unidecode(b)

    if remove_punctuation(a.lower()) == remove_punctuation(b.lower()):
        return True

    if grapheme_to_phoneme(a, separator='|') == grapheme_to_phoneme(b, separator='|'):
        return True

    return False


def gcs_uri_to_blob(gcs_uri):
    """ Parse GCS URI and return a `Blob`.

    NOTE: This function requires GCS authorization.

    Args:
        gcs_uri (str): URI to a GCS object (e.g. "gs://cloud-samples-tests/speech/brooklyn.flac")

    Returns:
        (google.cloud.storage.blob.Blob)
    """
    assert len(gcs_uri) > 5, 'The URI must be longer than 5 characters to be a valid GCS link.'
    assert gcs_uri[:5] == 'gs://', 'The URI provided is not a valid GCS link.'
    path_segments = gcs_uri[5:].split('/')
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(path_segments[0])
    blob = bucket.get_blob('/'.join(path_segments[1:]))
    return blob


def blob_to_gcs_uri(blob):
    """
    Args:
        blob (google.cloud.storage.blob.Blob)

    Returns:
        (str): URI to a GCS object (e.g. "gs://cloud-samples-tests/speech/brooklyn.flac")
    """
    return 'gs://' + blob.bucket.name + '/' + blob.name


def _format_gap(scripts, tokens, stt_tokens):
    """ Format the span of `tokens` and `stt_tokens` that lie between two alignments, the gap.
    """
    minus = lambda t: '\n' + COLORS['red'] + '--- "' + t + '"' + COLORS['reset_all'] + '\n'
    if len(tokens) > 2 or len(stt_tokens) > 0:
        if tokens[0].script_index != tokens[-1].script_index:
            yield minus(scripts[tokens[0].script_index][tokens[0].end_text:])
            yield '=' * 100
            for i in range(tokens[0].script_index + 1, tokens[-1].script_index):
                yield minus(scripts[i])
                yield '=' * 100
            yield minus(scripts[tokens[-1].script_index][:tokens[-1].start_text])
        else:
            yield minus(scripts[tokens[0].script_index][tokens[0].end_text:tokens[-1].start_text])
        text = ' '.join([t.text for t in stt_tokens])
        yield COLORS['green'] + '+++ "' + text + '"' + COLORS['reset_all'] + '\n'
    elif len(tokens) == 2 and tokens[0].script_index == tokens[-1].script_index:
        yield scripts[tokens[0].script_index][tokens[0].end_text:tokens[-1].start_text]
    elif len(tokens) == 2:
        yield scripts[tokens[0].script_index][tokens[0].end_text:]
        yield '\n' + '=' * 100 + '\n'
        yield scripts[tokens[-1].script_index][:tokens[-1].start_text]


def format_differences(scripts, alignments, tokens, stt_tokens):
    """ Format the differences between `tokens` and `stt_tokens` given the `alignments`.

    Args:
        scripts (list of str): The voice-over scripts.
        alignments (list of tuple): Alignments between `tokens` and `stt_tokens`.
        tokens (list of ScriptToken): Tokens derived from `scripts`.
        stt_tokens (list of SttToken): The predicted speech-to-text tokens.

    Returns:
        (generator): Generates `str` snippets that can be joined together for the full output.

    Example Output:
    > Welcome to Morton Arboretum, home to more than
    > --- "36 HUNDRED "
    > +++ "3,600"
    > native trees, shrubs, and plants.
    """
    tokens = ([ScriptToken(end_text=0, script_index=0)] + tokens +
              [ScriptToken(start_text=len(scripts[-1]), script_index=len(scripts) - 1)])
    alignments = [(a + 1, b) for a, b in alignments]

    groups = [(i, list(g)) for i, g in groupby(alignments, lambda a: tokens[a[0]].script_index)]
    groups = [(None, [(0, -1)])] + groups + [(None, [(len(tokens) - 1, len(stt_tokens))])]

    current = groups[0][1]
    for (_, before), (i, current), (_, after) in zip(groups, groups[1:], groups[2:]):
        is_unaligned = zip([before[-1]] + current, current + [after[0]])
        if any([b[0] - a[0] > 1 or b[1] - a[1] > 1 for a, b in is_unaligned]):
            yield from _format_gap(scripts, tokens[before[-1][0]:current[0][0] + 1],
                                   stt_tokens[before[-1][1] + 1:current[0][1]])
            for last, next_ in zip(current, current[1:] + [None]):
                yield scripts[i][tokens[last[0]].start_text:tokens[last[0]].end_text]
                if next_:
                    yield from _format_gap(scripts, tokens[last[0]:next_[0] + 1],
                                           stt_tokens[last[1] + 1:next_[1]])

    yield from _format_gap(scripts, tokens[current[-1][0]:], stt_tokens[current[-1][1] + 1:])


def align_stt_with_script(scripts,
                          stt_result,
                          get_alignment_window=lambda a, b: max(round(a * 0.1), 256) + abs(a - b)):
    """ Align an STT result(s) with the related script(s).

    NOTE:
      - There are a number of alignment issues due to dashes. Google's STT predictions around
        dashes are inconsistent with the original script. Furthermore, we cannot remove the dashes
        prior to alignment because Google's STT predictions do not provide timing for the words
        seperated by dashed. Lastly, Google does not have an option to remove dashes, either.
      - This approach for alignment is accurate but relies on white space tokenization. With white
        space tokenization, it's easy to reconstruct text spans; however, this approach messes up in
        cases like "parents...and", "tax-deferred", "little-c/Big-C" and "1978.In".
      - The alignment window default was set based off these assumptions:
        - The 10% window has performed well based on empirical evidence while a 5% window
          did not; furthermore, a 10% window is justified assuming a 6 - 10% insertion and deletion
          rate following an alignment.
        - The `max(x, 256)` is present for small sequences. 256 is small enough that it won't
          cause performance issues.
        - The `+ abs(len(script_tokens) - len(stt_tokens))` is for strange cases that
          include large misaligned sections.

    TODO:
      - Consider this approach:
          1. Align the STT transcript to the original transcript without sentence-level punctuation
            avoiding the error cases mentioned above.
          2. Align pervious alignment with the white space tokenized original transcript tokens.
            This approach should succeed to align "1978.In" and "parents...and"; however, it would
            still fail on "tax-deferred" because Google's STT will still include dashes in their
            output.
      - Adding visualization tools to help visualize the remaining errors:
        - STT and script have different character sets
        - STT and script have different token counts
        - STT and script didn't align well, and required lots of insertions or deletions.
        - There were tokens that could have aligned but were slightly different or there were tokens
          that aligned that were dramatically different.
        - STT didn't align with the audio correctly, and the timing is off.
        - The longest unaligned sequences.
        - The % of words that were aligned.
        The goal of this script is not to fix these issues, I think. We can fix some of these issues
        in the main training script; however, we do have an opprotunity to fix the transcript...
        or to manually adjust the alignment. We also can test other STT softwares.

    Args:
        scripts (list of str): The voice-over script.
        stt_result (list): The speech-to-text results for the related voice-over.
        get_alignment_window (callable, optional): Callable for computing the maximum window size in
            tokens for aligning the STT results with the provided script.

    Returns: TODO
    """
    # Tokenize tokens similar to Google STT for alignment.
    # NOTE: Google STT as of June 21st, 2019 tends to tokenize based on white spaces.
    script_tokens = flatten(
        [[ScriptToken(m.group(0), m.start(), m.end(), i)
          for m in re.finditer(r'\S+', script)]
         for i, script in enumerate(scripts)])
    stt_tokens = flatten([[
        SttToken(
            text=w['word'],
            start_audio=float(w['startTime'][:-1]),
            end_audio=float(w['endTime'][:-1],)) for w in r['words']
    ] for r in stt_result])
    # TODO: Assert that `stt_tokens` are white space deliminated

    num_unaligned, alignments = align_tokens([t.text for t in script_tokens],
                                             [t.text for t in stt_tokens],
                                             get_alignment_window(
                                                 len(script_tokens), len(stt_tokens)),
                                             allow_substitution=is_sound_alike)
    logger.info('Total word(s) unaligned: %s',
                format_ratio(len(script_tokens) - len(alignments), len(script_tokens)))
    logger.info('Failed to align: %s',
                format_differences(scripts, alignments, script_tokens, stt_tokens))

    return_ = [(script_tokens[a[0]].script_index, (script_tokens[a[0]].start_text,
                                                   script_tokens[a[0]].end_text),
                (stt_tokens[a[1]].start_audio, stt_tokens[a[1]].end_audio)) for a in alignments]
    return_ = [[(i[1], i[2]) for i in g] for _, g in groupby(return_, lambda i: i[0])]
    assert len(return_) == len(scripts), 'Invariant failure.'
    return return_


def _get_speech_context(script, max_phrase_length=100):
    """ Given the voice-over script generate `SpeechContext` to help Speech-to-Text recognize
    specific words or phrases more frequently.

    TODO:
      - Should we normalize the phrases with `unidecode.unidecode` and removing punctuation?
      - Should we pass a word or phrase distribution with the `boost` feature? (phrase)
      - Should the phrases be longer or shorter? (longer)
      - Should we boost the phrases? (likely not)
      - Does speech context help? (yes)

    Args:
        script (str)
        max_phrase_length (int, optional): The maximum character length of a phrase.
        boost (int, optional): See
        https://googleapis.dev/python/speech/latest/gapic/v1p1beta1/types.html#google.cloud.speech_v1p1beta1.types.SpeechContext

    Returns:
        (list of str)

    Example:
        >>> _get_speech_context('a b c d e f g h i j', 5)
        phrases: "j"
        phrases: "a b c"
        phrases: "g h i"
        phrases: "d e f"

    """
    # Tokenize tokens similar to Google STT for alignment.
    # NOTE: Google STT as of June 21st, 2019 tends to tokenize based on white spaces.
    spans = [(m.start(), m.end()) for m in re.finditer(r'\S+', script)]
    phrases = []
    start = 0
    end = 0
    for span in spans:
        if span[1] - start <= max_phrase_length:
            end = span[1]
        else:
            phrases.append(script[start:end])
            start = span[0]
    phrases.append(script[start:])
    return types.SpeechContext(phrases=list(set(phrases)))


def run_stt(audio_blobs,
            scripts,
            dest_blobs,
            max_stt_requests_per_second=50,
            stt_config=types.RecognitionConfig(
                language_code='en-US',
                model='video',
                use_enhanced=True,
                enable_automatic_punctuation=True,
                enable_word_time_offsets=True)):
    """ Run speech-to-text on `audio_blobs` and save them at `dest_blobs`.

    NOTE:
    - The documentation for `types.RecognitionConfig` does not include all the available
      options. All the potential parameters are included here:
      https://cloud.google.com/speech-to-text/docs/reference/rest/v1p1beta1/RecognitionConfig
    - Time offset values are only included for the first alternative provided in the recognition
      response, learn more here: https://cloud.google.com/speech-to-text/docs/async-time-offsets
    - Careful not to exceed default 300 requests a second quota; however, if you end up exceeding
      the quota, Google will force you to wait up to 30 minutes before it allows more requests.
    - The `stt_config` defaults are set based on:
      https://blog.timbunce.org/2018/05/15/a-comparison-of-automatic-speech-recognition-asr-systems/
    - You can preview the running operations via the command line. For example:
      $ gcloud ml speech operations describe 5187645621111131232

    TODO: Investigate using alternatives text for alignment.
    TODO: Investigate adding additional words to Google's STT vocab, and including our expected
    word distribution or n-gram distribution.
    TODO: Investigate updating `types.RecognitionConfig` with a better model.
    TODO: Update the return value for this function.
    TODO: Make the above arguments configurable.
    TODO: `max_stt_requests_per_second` is only considered every iteration; however, it's not
    guarenteed that every iteration has only one request to STT.

    Args:
        audio_blobs (list of google.cloud.storage.blob.Blob): List of GCS voice-over blobs.
        scripts (list of str): List of voice-over scripts.
        dest_blobs (list of google.cloud.storage.blob.Blob): List of GCS blobs to upload
            results too.
        max_stt_requests_per_second (int, optional): The maximum requests per second to make to STT.
        stt_config (types.RecognitionConfig, optional)

    Returns: None
    """
    operations = []
    for audio_blob, script, dest_blob in zip(audio_blobs, scripts, dest_blobs):
        config = copy(stt_config)
        config.speech_contexts.append(_get_speech_context('\n'.join(script)))
        audio = types.RecognitionAudio(uri=blob_to_gcs_uri(audio_blob))
        operations.append(speech.SpeechClient().long_running_recognize(config, audio))
        logger.info('STT operation %s "%s" started.', operations[-1].operation.name, dest_blob.name)

    progress = [0] * len(operations)
    progress_bar = tqdm(total=100)
    while not all(o is None for o in operations):
        for i, operation in enumerate(operations):
            if operation is None:
                continue

            metadata = operation.metadata
            if operation.done():
                dest_blobs[i].upload_from_string(json.dumps(MessageToDict(operation.result())))
                logger.info('STT operation %s "%s" finished.', operation.operation.name,
                            dest_blobs[i].name)
                operations[i] = None
                progress[i] = 100
            elif metadata is not None and metadata.progress_percent is not None:
                progress[i] = metadata.progress_percent
            progress_bar.update(min(progress) - progress_bar.n)
            time.sleep(1 / max_stt_requests_per_second)


def natural_keys(text):
    """ Returns keys (`list`) for sorting in a "natural" order.

    Inspired by: http://nedbatchelder.com/blog/200712/human_sorting.html
    """
    return [(int(char) if char.isdigit() else char) for char in re.split(r'(\d+)', str(text))]


def main(gcs_voice_overs,
         gcs_scripts,
         gcs_dests,
         text_column='Content',
         stt_folder='speech_to_text_results/',
         alignments_folder='alignments/'):
    """ Align `gcs_scripts` to `gcs_voice_overs` and cache the results at `gcs_dest`.

    Args:
        gcs_voice_overs (list of str): List of GCS URI(s) to voice-over(s).
        gcs_scripts (list of str): List of GCS URI(s) to voice-over script(s). The script(s) are
            stored in CSV format with related metadata.
        gcs_dests (list of str): GCS location(s) to upload results too.
        text_column (str, optional): The voice-over script column in the CSV script files.
        stt_folder (str, optional): The name of the folder to save speech-to-text results to.
        alignments_folder (str, optional): The name of the folder to save alignments too.
    """
    recorder = RecordStandardStreams().start()  # Save a log of the execution for future reference

    set_hparams()

    dest_blobs = [gcs_uri_to_blob(d) for d in gcs_dests]
    audio_blobs = [gcs_uri_to_blob(v) for v in gcs_voice_overs]

    filenames = [b.name.split('/')[-1].split('.')[0] + '.json' for b in audio_blobs]
    stt_blobs = [b.bucket.blob(b.name + stt_folder + n) for b, n in zip(dest_blobs, filenames)]
    alignment_blobs = [
        b.bucket.blob(b.name + alignments_folder + n) for b, n in zip(dest_blobs, filenames)
    ]

    logger.info('Downloading voice-over scripts...')
    scripts = [gcs_uri_to_blob(s).download_as_string().decode('utf-8') for s in gcs_scripts]
    scripts = [pandas.read_csv(StringIO(s))[text_column].tolist() for s in scripts]

    logger.info('Running speech-to-text and caching results...')
    filtered = list(filter(lambda i: not i[-1].exists(), zip(audio_blobs, scripts, stt_blobs)))
    if len(filtered) > 0:
        run_stt(*zip(*filtered))
    stt_results = [json.loads(b.download_as_string()) for b in stt_blobs]
    stt_results = [s['results'] for s in stt_results]
    stt_results = [[r['alternatives'][0] for r in s if r['alternatives'][0]] for s in stt_results]

    for script, stt_result, alignment_blob in zip(scripts, stt_results, alignment_blobs):
        logger.info('Running alignment "%s" and uploading results...', alignment_blob.name)
        alignment = align_stt_with_script(script, stt_result)
        alignment_blob.upload_from_string(json.dumps(alignment))

    logger.info('Uploading logs...')
    for dest_blob in set(dest_blobs):
        blob = dest_blob.bucket.blob(dest_blob.name + recorder.log_path.name)
        blob.upload_from_string(recorder.log_path.read_text())


if __name__ == "__main__":  # pragma: no cover
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--voice_over', nargs='+', type=str, help='GCS link(s) to audio file(s).')
    parser.add_argument(
        '--voice_over_script', nargs='+', type=str, help='GCS link(s) to the script(s).')
    parser.add_argument(
        '--destination',
        nargs='+',
        type=str,
        help='GCS location(s) to upload alignment(s) and speech-to-text result(s).')
    parser.add_argument(
        '--sort', action='store_true', help='Sort the script, audio and destination lists.')
    args = parser.parse_args()

    if args.sort:
        args.voice_over = sorted(args.voice_over, key=natural_keys)
        args.voice_over_script = sorted(args.voice_over_script, key=natural_keys)
        args.destination = sorted(args.destination, key=natural_keys)

    if len(args.voice_over) > len(args.destination):
        ratio = len(args.voice_over) // len(args.destination)
        args.destination = list(chain.from_iterable(repeat(x, ratio) for x in args.destination))

    main(args.voice_over, args.voice_over_script, args.destination)

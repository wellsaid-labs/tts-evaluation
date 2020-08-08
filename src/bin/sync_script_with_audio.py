""" Script for generating an alignment between a script and a voice-over.

Given a audio file with a voice-over of some text, this script generate and cache an alignment
from text spans to audio spans. Furthermore, this script will help resolve any issues where the
script or audio is missing or has extra data.

Prior:

    Setup a service account for your local machine that allows you to write to the
    `wellsaid_labs_datasets` bucket in Google Cloud Storage. Start by creating a service account
    using the Google Cloud Console. Give the service account the
    "Google Cloud Storage Object Admin" role (f.y.i. the "Role" menu scrolls). Name the service
    account something similar to "michael-gcs-datasets". Download a key as JSON and put it
    somewhere secure locally:

    $ mv ~/Downloads/voice-research-255602-1a5538456fc3.json gcs_credentials.json

Example:

    GOOGLE_APPLICATION_CREDENTIALS=gcs_credentials.json \
    python -m src.bin.sync_script_with_audio --audio 'gs://wellsaid_labs_datasets/hilary_noriega/preprocessed_recordings/Script 1.wav' \
                                             --script 'gs://wellsaid_labs_datasets/hilary_noriega/scripts/Script 1 - Hilary.csv' \
                                             --destination gs://wellsaid_labs_datasets/hilary_noriega/

Example (Batch):

    PREFIX=gs://wellsaid_labs_datasets/hilary_noriega
    GOOGLE_APPLICATION_CREDENTIALS=gcs_credentials.json \
    python -m src.bin.sync_script_with_audio \
      --audio "${(@f)$(gsutil ls "$PREFIX/preprocessed_recordings/*.wav")}" \
      --script "${(@f)$(gsutil ls "$PREFIX/scripts/*.csv")}" \
      --destination $PREFIX/

"""
from collections import namedtuple
from copy import copy
from functools import lru_cache
from io import StringIO

import argparse
import json
import logging
import re
import time

import pandas
import unidecode

from google.cloud import speech
from google.cloud import storage
from google.cloud.speech import types
from google.protobuf.json_format import MessageToDict
from tqdm import tqdm

from src.environment import COLORS
from src.environment import set_basic_logging_config
from src.environment import TEMP_PATH
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
ScriptToken = namedtuple('ScriptToken', ['text', 'start_text', 'end_text', 'script_index'])

# Args:
#   text (str): The script text.
#   start_audio (float): The beginning of the audio span in seconds during which `text` is voiced.
#   end_audio (float): The end of the audio span in seconds during which `text` is voiced.
SttToken = namedtuple('SttToken', ['text', 'start_audio', 'end_audio'])


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


def print_differences(scripts, alignments, tokens, stt_tokens):
    """ Print the differences between `tokens` and `stt_tokens` given the `alignments`.

    Args:
        scripts (list of str): The voice-over scripts.
        alignments (list of tuple): Alignments between `tokens` and `stt_tokens`.
        tokens (list of ScriptToken): Tokens derived from `scripts`.
        stt_tokens (list of SttToken): The predicted speech-to-text tokens.

    Example Output:
    > Welcome to Morton Arboretum, home to more than
    > --- "36 HUNDRED "
    > +++ "3,600"
    > native trees, shrubs, and plants.
    """
    minuses = lambda span: print(COLORS['red'], '--- "' + span + '"' + COLORS['reset_all'])
    pluses = lambda span: print(COLORS['green'], '+++ "' + span + '"' + COLORS['reset_all'])

    pairs = zip([(0, 0)] + alignments, alignments + [(len(tokens) - 1, len(stt_tokens) - 1)])
    for last, next in pairs:
        last_script_index = tokens[last[0]].script_index
        next_script_index = tokens[next[0]].script_index
        last_script = scripts[last_script_index]
        next_script = scripts[next_script_index]

        if next[0] - last[0] > 1 or next[1] - last[1] > 1:
            print(last_script[tokens[last[0]].start_text:tokens[last[0]].end_text])
            if last_script_index != next_script_index:
                print('-' * 100)
                minuses(last_script[tokens[last[0]].end_text + 1:])
                minuses(next_script[:tokens[next[0]].start_text])
            else:
                minuses(last_script[tokens[last[0]].end_text + 1:tokens[next[0]].start_text])
            pluses(' '.join([t.text for t in stt_tokens[last[1] + 1:next[1]]]))
            if last_script_index != next_script_index:
                print('-' * 100)
            print(next_script[tokens[next[0]].start_text:tokens[next[0]].start_text], end='')
        elif last_script_index != next_script_index:
            print(last_script[tokens[last[0]].start_text:])
            print('=' * 100)
            print(next_script[:tokens[next[0]].start_text], end='')
        else:
            span = last_script[tokens[last[0]].start_text:tokens[next[0]].start_text]
            print(span, end='')


def align_stt_with_script(
    scripts,
    stt_result,
    get_alignment_window=lambda a, b: max(round(a * 0.1), 256) + abs(a - b),
    allow_substitution=is_sound_alike,
):
    """ Align an STT results with the script.

    NOTE:
      - There are a number of alignment issues due to dashes. Google's STT predictions around
        dashes are inconsistent with the original script. Furthermore, we cannot remove the dashes
        prior to alignment because Google's STT predictions do not provide timing for the words
        seperated by dashed. Lastly, Google does not have an option to remove dashes, either.
      - This approach for alignment is accurate but relies on white space tokenization. With white
        space tokenization, it's easy to reconstruct text spans; however, this approach messes up in
        cases like "parents...and", "tax-deferred", "little-c/Big-C" and "1978.In".
      - The alignment window default was set based off these assumptions:
        - The 10% window has performed well based off empirical evidence while a 5% window
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
      - Allow substitutions iff the text is in the "alternatives" list.
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
        get_alignment_window (lambda, optional): Callable for computing the maximum window size in
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

    logger.info('Aligning transcript with STT...')
    num_unaligned, alignments = align_tokens([t.text for t in script_tokens],
                                             [t.text for t in stt_tokens],
                                             get_alignment_window(
                                                 len(script_tokens), len(stt_tokens)),
                                             allow_substitution=allow_substitution)
    logger.info('Total word(s) unaligned: %s',
                format_ratio(len(script_tokens) - len(alignments), len(script_tokens)))
    print_differences(scripts, alignments, script_tokens, stt_tokens)

    # TODO: Save the results in a alignment folder in the actor's bucket
    # TODO: Save the results so that they are aligned the scripts to the audio.
    return [((script_tokens[a[0]].start_text, script_tokens[a[0]].end_text),
             (stt_tokens[a[1]].start_audio, stt_tokens[a[1]].end_audio)) for a in alignments]


def _get_speech_context(script, max_phrase_length=100):
    """ Given the voice-over script generate `SpeechContext` to help Speech-to-Text recognize
    specific words or phrases more frequently.

    TODO:
      - Should we normalize the phrases with `unidecode.unidecode` and removing punctuation?
      - Should we pass a word or phrase distribution with the `boost` feature?
      - Should the phrases be longer or shorter?

    Args:
        script (str)
        max_phrase_length (int)

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


def _get_stt(audio_blobs, scripts, dest_blobs, stt_config, max_stt_requests_per_second):
    """ Run speech-to-text on `audio_blobs` and save them at `dest_blobs`.

    NOTE: You can preview the running operations via the command line. For example:
    $ gcloud ml speech operations describe 5187645621111131232

    TODO: `max_stt_requests_per_second` is only considered every iteration; however, it's not
    guarenteed that every iteration has only one request to STT.

    Args:
        audio_blobs (list of google.cloud.storage.blob.Blob)
        scripts (list of str)
        dest_blobs (list of google.cloud.storage.blob.Blob)
        stt_config (types.RecognitionConfig)
        max_stt_requests_per_second (int)

    Returns: None
    """
    operations = []
    for audio_blob, script, dest_blob in zip(audio_blobs, scripts, dest_blobs):
        config = copy(stt_config)
        config.speech_contexts.append(_get_speech_context('\n'.join(script)))
        audio = types.RecognitionAudio(uri=blob_to_gcs_uri(audio_blob))
        operations.append(speech.SpeechClient().long_running_recognize(config, audio))
        logger.info('Operation %s "%s" started.', operations[-1].operation.name, dest_blob.name)

    progress = [0] * len(operations)
    total_size = sum([a.size for a in audio_blobs])
    progress_bar = tqdm(total=100)
    while not all(o is None for o in operations):
        for i, operation in enumerate(operations):
            if operation is None:
                continue

            metadata = operation.metadata
            if operation.done():
                dest_blob.upload_from_string(json.dumps(MessageToDict(operation.result())))
                logger.info('Operation %s "%s" done.', operation.operation.name, dest_blobs[i].name)
                operations[i] = None
                progress[i] = 100 * audio_blobs[i].size
            elif metadata is not None and metadata.progress_percent is not None:
                progress[i] = metadata.progress_percent * audio_blobs[i].size
            progress_bar.update(round(sum(progress) / total_size) - progress_bar.n)

            time.sleep(1 / max_stt_requests_per_second)


def get_stt(audio_blobs,
            scripts,
            dest_blob,
            max_stt_requests_per_second=50,
            stt_config=types.RecognitionConfig(
                language_code='en-US',
                model='video',
                use_enhanced=True,
                enable_automatic_punctuation=True,
                enable_word_time_offsets=True)):
    """ Get Google speech-to-text (STT) results for each audio file in `gcs_audio_files`.

    NOTE:
    - The documentation for `types.RecognitionConfig` does not include all the available
      options. All the potential parameters are included here:
      https://cloud.google.com/speech-to-text/docs/reference/rest/v1p1beta1/RecognitionConfig
    - Time offset values are only included for the first alternative provided in the recognition
      response, learn more here: https://cloud.google.com/speech-to-text/docs/async-time-offsets
    - The results will be cached in `dest_blob` with the same filename as the audio file
      under the 'speech-to-text' folder.
    - Careful not to exceed default 300 requests a second quota; however, if you end up exceeding
      the quota, Google will force you to wait up to 30 minutes before it allows more requests.
    - The `stt_config` defaults are set based on:
    https://blog.timbunce.org/2018/05/15/a-comparison-of-automatic-speech-recognition-asr-systems/

    TODO: Investigate using alternatives text for alignment.
    TODO: Investigate adding additional words to Google's STT vocab, and including our expected
    word distribution or n-gram distribution.
    TODO: Investigate updating `types.RecognitionConfig` with a better model.
    TODO: Update the return value for this function.
    TODO: Make the above arguments configurable.

    Args:
        audio_blobs (list of google.cloud.storage.blob.Blob): List of GCS audio blobs.
        scripts (list of str): List of scripts.
        dest_blob (google.cloud.storage.blob.Blob): GCS blob to upload results too.
        max_stt_requests_per_second (int): The maximum requests per second to make to STT.
        stt_config (types.RecognitionConfig)

    Returns:
        (list) [
          {
            script (str): Script of the audio chunk.
            confidence (float): Google's confidence in the script.
            words (list): List of words in the script tokenized by spaces.
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
    names = ['speech_to_text/' + p.name.split('/')[-1].split('.')[0] + '.json' for p in audio_blobs]
    assert len(
        set(names)) == len(names), 'This function requires that all audio files have unique names.'
    dest_blobs = [dest_blob.bucket.blob(dest_blob.name + n) for n in names]

    # Run speech-to-text on audio files iff their results are not found.
    filtered = [(a, s, b) for a, s, b in zip(audio_blobs, scripts, dest_blobs) if not b.exists()]
    if len(filtered) > 0:
        _get_stt(*zip(*filtered), stt_config, max_stt_requests_per_second)

    # Get all speech-to-text results.
    return [json.loads(b.download_as_string()) for b in dest_blobs]


def normalize_text(text):
    """ Normalize the text such that the text matches up closely to the audio.

    NOTE: These rules are specific to the datasets processed so far.
    TODO: Remove from this script...

    Args:
        text (str)

    Returns
        text (str)
    """
    text = text.strip()
    text = text.replace('\t', '  ')
    text = text.replace('®', '')
    text = text.replace('™', '')
    # Remove HTML tags
    text = re.sub('<.*?>', '', text)
    # Fix for a missing space between end and beginning of a sentence.
    # Example match deliminated by <>:
    #   the cold w<ar.T>he term 'business ethics'
    text = re.sub(r"([a-z]{2}[.!?])([A-Z])", r"\1 \2", text)
    return unidecode.unidecode(text)


def main(gcs_audio_files,
         gcs_script_files,
         gcs_dest,
         text_column='Content',
         tmp_file_path=TEMP_PATH):
    """ Align the script to the audio file and cache the results.

    TODO: Save the alignment.

    Args:
        gcs_audio_files (list of str): List of GCS URIs to audio files.
        gcs_script_files (list of str): List of GCS URIs to voice-over scripts. The scripts
            are stored in CSV format.
        gcs_dest (str): GCS location to upload results too.
        text_column (str, optional): The voice-over script column in the CSV script files.
        tmp_file_path (pathlib.Path, optional)
    """
    # TODO: Save the logs to the bucket.
    # TODO: Naturally sort the audio files and script files, unless gsutil already does this...
    RecordStandardStreams(tmp_file_path).start()  # Save a log of the execution for future reference

    set_hparams()

    dest_blob = gcs_uri_to_blob(gcs_dest)
    audio_blobs = [gcs_uri_to_blob(a) for a in gcs_audio_files]
    script_data_frames = [
        pandas.read_csv(StringIO(gcs_uri_to_blob(s).download_as_string().decode('utf-8')))
        for s in gcs_script_files
    ]
    script_texts = [
        [normalize_text(t) for t in df[text_column].tolist()] for df in script_data_frames
    ]

    stt_results = get_stt(audio_blobs, script_texts, dest_blob)

    # Preprocess `stt_results` removing unnecessary keys or hierarchy.
    stt_results = [
        [r['alternatives'][0] for r in s['results'] if r['alternatives'][0]] for s in stt_results
    ]

    for script_text, stt_result in zip(script_texts, stt_results):
        alignment = align_stt_with_script(script_text, stt_result)


if __name__ == "__main__":  # pragma: no cover
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--audio', nargs='+', type=str, help='GCS link(s) to audio file(s).')
    parser.add_argument('--script', nargs='+', type=str, help='GCS link(s) to the script(s).')
    parser.add_argument(
        '--destination',
        type=str,
        help='GCS location to upload alignments and speech-to-text results.')
    args = parser.parse_args()
    main(args.audio, args.script, args.destination)

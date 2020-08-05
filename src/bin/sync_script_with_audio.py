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
from io import StringIO
from tqdm import tqdm

import argparse
import json
import logging
import re
import time

import pandas

from google.api_core.retry import Retry
from google.cloud import speech
from google.cloud import storage
from google.cloud.speech import types
from google.protobuf.json_format import MessageToDict

from src.environment import TEMP_PATH
from src.hparams import set_hparams
from src.utils import align_tokens
from src.utils import flatten
from src.utils import RecordStandardStreams
from src.environment import set_basic_logging_config

set_basic_logging_config()
logger = logging.getLogger(__name__)


def gcs_uri_to_blob(gcs_uri):
    """ Parse GCS URI and return a `Blob`.

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
    return bucket.blob('/'.join(path_segments[1:]))


def align_stt_with_script(scripts,
                          stt_result,
                          get_alignment_window=lambda a, b: max(round(a * 0.1), 256) + abs(a - b)):
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

    Returns:
        (list of Alignment or Nonalignment): List of scripts and their corresponding token
            alignments.
    """
    # Tokenize tokens similar to Google STT for alignment.
    # NOTE: Google STT as of June 21st, 2019 tends to tokenize based on white spaces.
    # NOTE: `.lower()` to assist in alignment in cases like "HEALTH" and "health".
    script_tokens = flatten([[{
        'text': m.group(0).lower(),
        'start_text': m.start(),
        'end_text': m.end(),
        'script_index': i
    } for m in re.finditer(r'\S+', script)] for i, script in enumerate(scripts)])
    stt_tokens = flatten([[{
        'text': w['word'].lower(),
        'start_audio': float(w['startTime'][:-1]),
        'end_audio': float(w['endTime'][:-1]),
    } for w in r['words']] for r in stt_result])

    logger.info('Aligning transcript with STT.')

    num_unaligned, alignments = align_tokens([t['text'] for t in script_tokens],
                                             [t['text'] for t in stt_tokens],
                                             get_alignment_window(
                                                 len(script_tokens), len(stt_tokens)))

    # TODO: Save the results in a alignment folder in the actor's bucket
    # TODO: Save the results so that they are aligned the scripts to the audio.
    return [((script_tokens[a[0]]['start_text'], script_tokens[a[0]]['end_text']),
             (stt_tokens[a[1]]['start_audio'], stt_tokens[a[1]]['end_audio'])) for a in alignments]


def _get_stt(gcs_audio_files, blobs, stt_config, max_requests_per_second):
    """ Run speech-to-text on `gcs_audio_files` and save them at `blobs`.
    """
    operations = [
        speech.SpeechClient().long_running_recognize(
            stt_config, types.RecognitionAudio(uri=f), retry=Retry())
        for f, b in zip(gcs_audio_files, blobs)
    ]
    blobs = list(blobs)
    progress = [0] * len(operations)
    # NOTE: You can preview the running operations via the command line. For example:
    # $ gcloud ml speech operations describe 5187645621111131232
    logger.info('Running %d speech-to-text operation(s):\n%s', len(operations),
                '\n'.join([str((o.operation.name, b.name)) for o, b in zip(operations, blobs)]))
    # TODO: This progress bar could be more accurate if it were weighted by file size (`blob.size`)
    # or audio length.
    with tqdm(total=100 * len(operations)) as progress_bar:
        while not all(o is None for o in operations):
            for i, (operation, blob) in enumerate(zip(operations, blobs)):
                if operation is not None and blob is not None:
                    metadata = operation.metadata
                    if operation.done():
                        blob.upload_from_string(json.dumps(MessageToDict(operation.result())))
                        logger.info('Operation %s "%s" is done.', operation.operation.name,
                                    blob.name)
                        blobs[i] = None
                        operations[i] = None
                        progress[i] = 100
                    elif metadata is not None and metadata.progress_percent is not None:
                        progress[i] = metadata.progress_percent
                    progress_bar.update(sum(progress) - progress_bar.n)
                    time.sleep(1 / max_requests_per_second)


def get_stt(gcs_audio_files,
            gcs_destination,
            max_requests_per_second=50,
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
    - The results will be cached in `gcs_destination` with the same filename as the audio file
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
        gcs_audio_files (list of str): List of GCS paths to audio files.
        max_requests_per_second (int): The maximum requests per second to make.
        gcs_destination (google.cloud.storage.bucket.Bucket): GCS location to upload results too.
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
    names = ['speech_to_text/' + p.split('/')[-1].split('.')[0] + '.json' for p in gcs_audio_files]
    assert len(
        set(names)) == len(names), 'This function requires that all audio files have unique names.'
    blobs = [gcs_destination.bucket.blob(gcs_destination.name + n) for n in names]

    # Run speech-to-text on audio files iff their results are not found.
    filtered = [(a, b) for a, b in zip(gcs_audio_files, blobs) if not b.exists()]
    if len(filtered) > 0:
        _get_stt(*zip(*filtered), stt_config, max_requests_per_second)

    # Get all speech-to-text results.
    return [json.loads(b.download_as_string()) for b in blobs]


def main(gcs_audio_files,
         gcs_script_files,
         gcs_destination,
         text_column='Content',
         tmp_file_path=TEMP_PATH):
    """ Align the script to the audio file and cache the results.

    TODO: Save the alignment.

    Args:
        gcs_audio_files (list of str): List of GCS URIs to audio files.
        gcs_script_files (list of str): List of GCS URIs to voice-over scripts. The scripts
            are stored in CSV format.
        gcs_destination (str): GCS location to upload results too.
        text_column (str, optional): The voice-over script column in the CSV script files.
        tmp_file_path (pathlib.Path, optional)
    """
    # TODO: Save the logs to the bucket.
    # TODO: Naturally sort the audio files and script files, unless gsutil already does this...
    RecordStandardStreams(tmp_file_path).start()  # Save a log of the execution for future reference

    set_hparams()

    gcs_destination = gcs_uri_to_blob(gcs_destination)
    stt_results = get_stt(gcs_audio_files, gcs_destination)

    # Preprocess `stt_results` removing unnecessary keys or hierarchy.
    stt_results = [
        [r['alternatives'][0] for r in s['results'] if r['alternatives'][0]] for s in stt_results
    ]

    for gcs_script_file, stt_result in zip(gcs_script_files, stt_results):
        script_file = gcs_uri_to_blob(gcs_script_file).download_as_string().decode('utf-8')
        data_frame = pandas.read_csv(StringIO(script_file))
        scripts = data_frame[text_column].tolist()
        alignment = align_stt_with_script(scripts, stt_result)
        logger.info('alignment: %s', alignment)


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

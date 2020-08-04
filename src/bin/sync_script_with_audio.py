""" Script for generating an alignment between a script and a voice-over.

Given a audio file with a voice-over of some text, this script generate and cache an alignment
from text spans to audio spans. Furthermore, this script will help resolve any issues where the
script or audio is missing or has extra data.


Questions:
- Should we add SoX preprocessing now? No. The SoX preprocessing can be run during runtime.
- The text should also be preprocessed during runtime.
- Can our voice actors directly upload the VO to a public GCS? How do we safely expose a GCS link
for the voice actor to upload to? Should we use a signed URL?
- Should we handle each file individually? Or as a group? (Group)
- How do we get all the files in a GCS bucket? (Not sure yet... We could just us `gsutil ls`)
- Can a GCS bucket be backed up? (Yes)

VO Upload Process:
1. Create a new bucket called "upload_{voice actor name}".
2. Go to permissions, and grant the voice actor permission to upload files.
3. Transfer the files to our secure bucket like "voice_over_data" that holds the data.

Alternative Process:
1. Have voice actors upload documents to Google Drive.
2. Download them, and reupload them to Google Storage like "voice_over_data".

Script:
1. Arguments: Public GCS link with script, public GCS link with VO, and GCS password
2. Download and read the script.
3. Run STT on GCS VO link (cache the STT)
4. Align the STT results with the script
5. QA the STT results
6. Upload the STT file to the GCS bucket
7. Upload an alignment file between STT and script

Prior:

    Setup a service account for your local machine that allows you to write to the
    `TODO` bucket in Google Cloud Storage. Start by creating a service account
    using the Google Cloud Console. Give the service account the
    "Google Cloud Storage Object Admin" role (f.y.i. the "Role" menu scrolls). Name the service
    account something similar to "michael-local-gcs". Download a key as JSON and put it somewhere
    secure locally:

    $ mv ~/Downloads/voice-research-255602-1a5538456fc3.json gcs_credentials.json

Example:

    GOOGLE_APPLICATION_CREDENTIALS=gcs_credentials.json \
    python -m src.bin.sync_script_with_audio --audio gs://bucket/audio.wav \
                                             --script gs://bucket/script.csv \
                                             --alignment gs://bucket/
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

from src.environment import COLORS
from src.environment import TEMP_PATH
from src.hparams import set_hparams
from src.utils import align_tokens
from src.utils import flatten
from src.utils import RecordStandardStreams

logging.basicConfig(
    format='{}%(levelname)s:%(name)s:{} %(message)s'.format(COLORS['light_magenta'],
                                                            COLORS['reset_all']),
    level=logging.INFO)
logger = logging.getLogger(__name__)


def gcs_uri_to_blob(gcs_uri):
    """ Parse GCS URI and return a `Blob`.

    Args:
        gcs_uri (str): URI to a GCS object (e.g. "gs://cloud-samples-tests/speech/brooklyn.flac")

    Returns:
        (google.cloud.storage.blob.Blob)
    """
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
        'start_audio': w['startTime'],
        'end_audio': w['endTime'],
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


def get_stt(gcs_audio_files,
            gcs_bucket,
            max_requests_per_second=100,
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
    - The results will be cached in `gcs_bucket` with the same filename as the audio file, and the
      extension `.stt`.
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
        gcs_bucket (google.cloud.storage.bucket.Bucket): GCS bucket to upload results too.
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
    # TODO: Save the results in a speech-to-text folder in the actor's bucket
    names = [p.split('/')[-1].split('.')[0] + '.stt' for p in gcs_audio_files]
    assert len(
        set(names)) == len(names), 'This function requires that all audio files have unique names.'
    blobs = [gcs_bucket.blob(n) for n in names]

    logger.info('Running speech-to-text for %d files', len(gcs_audio_files))
    operations = [
        speech.SpeechClient().long_running_recognize(
            stt_config, types.RecognitionAudio(uri=f), retry=Retry())
        for f, b in zip(gcs_audio_files, blobs)
        if not b.exists()
    ]
    progress = [0] * len(operations)
    with tqdm(total=100 * len(operations)) as progress_bar:
        while not all(o.done() for o in operations):
            for i, operation in enumerate(operations):
                metadata = operation.metadata()
                if metadata is not None:
                    metadata = MessageToDict(metadata)
                    if 'progressPercent' in metadata:
                        progress[i] = metadata['progressPercent']
                        progress_bar.update(sum(progress) - progress_bar.n)
                time.sleep(1 / max_requests_per_second)

    responses = [json.dumps(MessageToDict(o.result())) for o in operations]
    for response, name in zip(responses, names):
        gcs_bucket.blob(name).upload_from_string(response)
    return [b.download_as_string() for b in blobs]


def main(gcs_audio_files,
         gcs_script_files,
         gcs_bucket,
         text_column='Content',
         tmp_file_path=TEMP_PATH):
    """ Align the script to the audio file and cache the results.

    TODO: Save the alignment.

    Args:
        gcs_audio_files (list of str): List of GCS paths to audio files.
        gcs_script_files (list of str): List of GCS scripts to audio files. The scripts
            are stored in CSV format.
        gcs_bucket (str): GCS bucket to upload results too.
        text_column (str, optional): The script column
        tmp_file_path (pathlib.Path, optional)

    Returns:
        (list of dict): The STT results for each audio file.
    """
    # TODO: Save the logs to the bucket.
    RecordStandardStreams(tmp_file_path).start()  # Save a log of the execution for future reference

    set_hparams()

    gcs_bucket = storage.Client().bucket(gcs_bucket)
    stt_results = get_stt(gcs_audio_files, gcs_bucket)

    # Preprocess `stt_results` removing unnecessary keys or hierarchy.
    stt_results = stt_results['results']
    stt_results = [r['alternatives'][0] for r in stt_results if r['alternatives'][0]]

    for gcs_script_file, stt_result in zip(gcs_script_files, stt_results):
        script_file = gcs_uri_to_blob(gcs_script_file).download_as_string()
        data_frame = pandas.read_csv(StringIO(script_file))
        scripts = data_frame[text_column].tolist()
        alignment = align_stt_with_script(scripts, stt_result)

    logger.log(alignment)


if __name__ == "__main__":  # pragma: no cover
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--audio', type=str, help='GCS link(s) to audio file(s).')
    parser.add_argument('--script', type=str, help='GCS link(s) to the script(s).')
    parser.add_argument(
        '--bucket', type=str, help='GCS bucket to upload alignments and speech-to-text results.')
    args = parser.parse_args()
    main(args.audio, args.script)

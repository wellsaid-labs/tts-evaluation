""" Script for chunking a transcript with the associated audio file.

NOTE: Instead of testing this script via unit tests, we check various invariants with asserts.
"""
from pathlib import Path

import argparse
import json
import logging
import re
import requests
import sys

from collections import Counter

import pandas
import librosa

from src.audio import read_audio
from src.utils import duplicate_stream

GENTLE_SUCCESS_CASE = 'success'
GENTLE_OOV_WORD = '<unk>'
TERMINAL_COLOR_RESET = '\033[0m'
TERMINAL_COLOR_RED = '\033[91m'
TERMINAL_COLOR_PURPLE = '\033[95m'

logging.basicConfig(
    format='{}%(levelname)s:%(name)s:{} %(message)s'.format(TERMINAL_COLOR_PURPLE,
                                                            TERMINAL_COLOR_RESET),
    level=logging.INFO)
logger = logging.getLogger(__name__)


class Alignment():
    """ An alignment that was determined confidently between characters and text.

    Args:
        start_audio (int): Start of the audio in samples.
        end_audio (int): End of audio in samples.
        start_text (int): Start of text in characters.
        end_text (int): End of text in characters.
        next_alignment (Alignment or None): Next alignement.
        last_alignment (Alignment or None): Last alignement.
    """

    def __init__(self,
                 start_audio,
                 end_audio,
                 start_text,
                 end_text,
                 next_alignment=None,
                 last_alignment=None):
        self.start_audio = start_audio
        self.end_audio = end_audio
        self.start_text = start_text
        self.end_text = end_text
        self.next_alignment = next_alignment
        self.last_alignment = next_alignment


class Nonalignment():
    """ An alignment that was determined confidently between characters and text.

    Args:
        start_text (int): Start of text in characters.
        end_text (int): End of text in characters.
        next_alignment (Alignment or None): Next alignement.
    """

    def __init__(self, start_text, end_text, next_alignment=None):
        self.start_text = start_text
        self.end_text = end_text
        self.next_alignment = next_alignment


def natural_keys(text):
    '''
    Sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [(int(c) if c.isdigit() else c) for c in re.split('(\d+)', str(text))]


def _review_gentle(response, transcript):
    """ Log warnings, check invariants, and compute statistics for the script user.

    Args:
        response (dict): Response of Gentle
        transcript (str): Transcript sent to Gentle.
    """
    # Check various invariants
    assert response['transcript'] == transcript, 'Failed transcript invariant.'
    assert all([
        transcript[w['startOffset']:w['endOffset']] == w['word'] for w in response['words']
    ]), 'Transcript must align with word character offsets.'

    # Print warnings
    unaligned_text = None
    last_unaligned_word_index = None
    for i, word in enumerate(response['words']):
        if 'alignedWord' in word and word['case'] == GENTLE_SUCCESS_CASE:
            continue

        # Group up unaligned words into text snippets for a cleaner log
        if last_unaligned_word_index and last_unaligned_word_index + 1 == i:
            unaligned_text['endOffset'] = word['endOffset']
            unaligned_text['cases'].add(word['case'])
            last_unaligned_word_index += 1
        else:
            if unaligned_text:
                unaligned_text['text'] = transcript[unaligned_text['startOffset']:unaligned_text[
                    'endOffset']]
                logger.warning('Unaligned text: %s', unaligned_text)

            unaligned_text = {
                'startOffset': word['startOffset'],
                'endOffset': word['endOffset'],
                'cases': set([word['case']])
            }
            last_unaligned_word_index = i

    # Print any remaining unaligned text
    if unaligned_text:
        unaligned_text['text'] = transcript[unaligned_text['startOffset']:unaligned_text[
            'endOffset']]
        logger.warning('Unaligned text: %s', unaligned_text)

    # Warn if aligned words do not match with transcript
    for word in response['words']:
        if 'alignedWord' not in word or word['alignedWord'] == GENTLE_OOV_WORD:
            continue

        normalized_word = word['word'].lower().replace('’', '\'')
        if word['alignedWord'] != normalized_word:
            logger.warning('``alignedWord`` does not match transcript ``word``: %s', word)

    # Compute statistics
    cases_counter = Counter([w['case'] for w in response['words']])
    if len(cases_counter) > 1 or cases_counter[GENTLE_SUCCESS_CASE] == 0:
        for case in cases_counter:
            if case == GENTLE_SUCCESS_CASE:
                continue

            percentage = cases_counter[case] / sum(cases_counter.values()) * 100
            logger.warning('%f%% (%d of %d) of words have case: %s', percentage,
                           cases_counter[case], sum(cases_counter.values()), case)

    oov_words = [
        w['word']
        for w in response['words']
        if 'alignedWord' in w and w['alignedWord'] == GENTLE_OOV_WORD
    ]
    if len(oov_words) > 0:
        logger.warning('%f%% out of vocabulary words',
                       (len(oov_words) / len(response['words']) * 100))
        logger.warning('Out of vocabulary words: %s', sorted(list(set(oov_words))))


def _request_gentle(wav_path,
                    transcript,
                    gentle_cache_directory,
                    hostname='localhost',
                    port=8765,
                    parameters=(('async', 'false'),),
                    wait_per_second_of_audio=0.5,
                    sample_rate=24000):
    """ Align an audio file with the trascript at a word granularity.

    Args:
        wav_path (Path): The audio file path.
        transcript (str): The text spoken in the audio.
        gentle_cache_directory (Path): Directory used to cache Gentle requests.
        hostname (str, optional): Hostname used by the gentle server.
        port (int, optional): Port used by the gentle server.
        parameters (tuple, optional): Dictionary or bytes to be sent in the query string for the
            :class:`Request`.
        wait_per_second_of_audio (float, optional): Give the server time of
            ``wait_per_second_of_audio`` per second of audio it has to process.
        sample_rate (int, optional): Sample rate of the audio.

    Returns:
        (dict) {
            transcript (str): The transcript passed in.
            words (list) [
                {
                    alignedWord (str): Normalized and aligned word.
                    word (str): Parallel unnormalized word from transcript.
                    case (str): One of two cases 'success' or 'not-found-in-audio'.
                    start (float): Time the speakers begins to speak ``word`` in seconds.
                    end (float): Time the speakers stops speaking ``word`` in seconds.
                    startOffset (int): The trascript character index of the first letter of word.
                    endOffset (int): The trascript character index plus one of the last letter of
                        word.
                }, ...
            ]
        }
    """
    assert gentle_cache_directory.is_dir()
    gentle_cache_filename = (gentle_cache_directory / wav_path.name).with_suffix('.json')
    if gentle_cache_filename.is_file():
        logger.info('Using cached gentle alignment %s', str(gentle_cache_filename))
        response = json.loads(gentle_cache_filename.read_text())
        _review_gentle(response, transcript)
        return response

    duration = librosa.get_duration(filename=str(wav_path), sr=sample_rate)
    url = 'http://{}:{}/transcriptions'.format(hostname, port)
    response = requests.post(
        url,
        params=parameters,
        files={
            'audio': wav_path.read_bytes(),
            'transcript': transcript.encode(),
        },
        timeout=duration * wait_per_second_of_audio)
    if response.status_code != 200:
        raise ValueError('Gentle ({}) returned bad response: {}'.format(url, response.status_code))

    response = response.json()

    # Delete `phones` from the json object
    words = []
    for word in response['words']:
        word.pop('phones', None)
        words.append(word)
    response['words'] = words

    _review_gentle(response, transcript)
    gentle_cache_filename.write_text(json.dumps(response, sort_keys=True, indent=2))
    return response


def align_wav_and_scripts(wav_path,
                          scripts,
                          gentle_cache_directory,
                          sample_rate,
                          min_spoken_word_timing=0.045):
    """ Align an audio file with the scripts spoken in the audio.

    Notes:
        - The returned alignments do not include the entire script. We only include verified
          and confident alignments.

    Args:
        wav_path (Path): The audio file path.
        scripts (list of str): The scripts spoken in the audio.
        sample_rate (int): Sample rate of the audio.
        gentle_cache_directory (Path): Directory used to cache Gentle requests.
        min_spoken_word_timing (float, optional): When Gentle messes up alignment, the
            timing of the characters per second is a strong signal. This sets a minimum for the
            number of characters per second that can be said for it to be a valid alignment.

    Returns:
        (list of Alignment): For every script, a list of found alignments.
    """
    seconds_to_samples = lambda seconds: int(round(seconds * sample_rate))
    transcript = '\n'.join(scripts)
    logger.info('Characters in transcript %s', sorted(list(set(transcript))))
    alignment = _request_gentle(
        wav_path, transcript, gentle_cache_directory, sample_rate=sample_rate)
    get_spoken_word_timing = lambda w: (w['end'] - w['start']) / (w['endOffset'] - w['startOffset'])

    # Align seperate scripts with the transcript alignment.
    aligned_words = alignment['words']
    transcript_start_offset = 0
    all_scripts_alignments = []
    for script in scripts:
        scripts_alignments = []
        transcript_end_offset = transcript_start_offset + len(script)
        while (len(aligned_words) > 0 and aligned_words[0]['endOffset'] <= transcript_end_offset):
            aligned_word = aligned_words.pop(0)
            # NOTE: ``GENTLE_OOV_WORD`` is considered a nonalignment because Gentle frequently
            # messes up aligning those words
            if (aligned_word['case'] != GENTLE_SUCCESS_CASE or
                    get_spoken_word_timing(aligned_word) < min_spoken_word_timing):
                scripts_alignments.append(
                    Nonalignment(
                        start_text=aligned_word['startOffset'] - transcript_start_offset,
                        end_text=aligned_word['endOffset'] - transcript_start_offset,
                    ))
            else:
                scripts_alignments.append(
                    Alignment(
                        start_text=aligned_word['startOffset'] - transcript_start_offset,
                        end_text=aligned_word['endOffset'] - transcript_start_offset,
                        start_audio=seconds_to_samples(aligned_word['start']),
                        end_audio=seconds_to_samples(aligned_word['end'])))

        for i in range(len(scripts_alignments) - 1):
            scripts_alignments[i].next_alignment = scripts_alignments[i + 1]

        for i in range(1, len(scripts_alignments)):
            scripts_alignments[i].last_alignment = scripts_alignments[i - 1]

        if len(scripts_alignments) == 0:
            logger.warning('Unable to align script %s', wav_path)
        else:
            all_scripts_alignments.append(scripts_alignments)

        transcript_start_offset += len(script) + 1  # Add plus one for ``\n``

    assert len(all_scripts_alignments) == len(scripts), 'Unable to align all scripts.'
    assert len(aligned_words) == 0, 'Unable to line up aligned words with scripts.'
    return all_scripts_alignments


def _review_chunk_alignments(script, chunks, spans):
    """ Check some invariants and log warnings for ``chunk_alignments``

    NOTE:
        - Refer to ``chunk_alignments`` to better understand arguments.

    Returns:
        (int): Number of characters in the script aligned.
        (int): Number of characters in the script.
    """
    for chunk in chunks:
        assert isinstance(chunk[0], Alignment)
        assert isinstance(chunk[-1], Alignment)

    for i in range(len(spans) - 1):
        assert spans[i]['text'][1] <= spans[i + 1]['text'][0]
        assert spans[i]['audio'][1] <= spans[i + 1]['audio'][0]

    num_unaligned_characters = len(script) - sum([s['text'][1] - s['text'][0] for s in spans])
    if num_unaligned_characters != 0:
        to_print = [TERMINAL_COLOR_RED] + list(script) + [TERMINAL_COLOR_RESET]
        for span in reversed(spans):  # Inserting messes up the index of later items
            to_print.insert(span['text'][1] + 1, TERMINAL_COLOR_RED)
            to_print.insert(span['text'][0] + 1, TERMINAL_COLOR_RESET)
        logger.warning('Unable to align %f%% (%d of %d) of script, see here - \n%s',
                       num_unaligned_characters / len(script) * 100, num_unaligned_characters,
                       len(script), ''.join(to_print))

    return num_unaligned_characters, len(script)


def average_silence_delimiter(alignment):
    """ Compute the average samples of silence per character

    Args:
        alignment (Alignment)

    Returns:
        (float): -1 if
    """
    if isinstance(alignment, Nonalignment) or isinstance(alignment.next_alignment, Nonalignment):
        # NOTE: We want to start / end with Alignment chunks; therefore, we cannot cut at
        # at Nonalignment.
        return -1.0

    return (alignment.next_alignment.start_audio - alignment.end_audio) / (
        alignment.next_alignment.start_text - alignment.end_text)


def chunk_alignments(alignments, script, max_chunk_length, delimiter=average_silence_delimiter):
    """ Chunk audio and text to be less than ``max_chunk_length``.

    Notes:
        - We chunk based on the largest silence between alignments. We assume that the returned
          chunks are independent because there exists silence between chunks.
        - Ideal chunk independence means that prior text does not affect the voice actors
          interpretation of that chunk of text.

    TODO:
        - Consider instead of chunking up an audio book, creating random training data cuts
          during training.
        - Consider using to add noise to our cuts instead of always cutting on the longest
          silence.
        - Write a more efficient algorithm that doesn't recompute the silences and
          chunks every iteration.
        - Consider deliminating on POS phrases instead of silence; therefore, not biasing
          the model towards smaller silences.

    Args:
        alignments (list of Alignment): List of alignments between text and audio.
        script (str): Transcript alignments refer too.
        max_chunk_length (float): Number of samples for the maximum chunk audio length.
        delimiter (callable, optional): Given an alignment returns -1 or a positive floating point
            number. A larger number is more likely to be used to split a sequence.

    Returns:
        spans (list of dict {
          text (tuple(int, int)): Span of text as measured in characters.
          audio (tuple(int, int)): Span of audio as measured in samples.
        }
        (int): Number of characters in the script aligned.
        (int): Number of characters in the script.
    """
    chunks = []
    # NOTE: We cannot end / start on a Nonalignment; therefore, we cut them off.
    max_chunk = alignments
    while len(max_chunk) > 0 and isinstance(max_chunk[0], Nonalignment):
        max_chunk = max_chunk[1:]
    while len(max_chunk) > 0 and isinstance(max_chunk[-1], Nonalignment):
        max_chunk = max_chunk[:-1]
    if len(max_chunk) == 0:
        return [], (len(script), len(script))

    # Chunk the max chunk until the max chunk is smaller than ``max_chunk_length``
    get_chunk_length = lambda chunk: chunk[-1].end_audio - chunk[0].start_audio
    max_chunk_length = get_chunk_length(max_chunk)
    while len(max_chunk) > 1 and max_chunk_length > max_chunk_length:
        max_silence_index, _ = max(enumerate(max_chunk[:-1]), key=lambda a: delimiter(a[1]))

        if delimiter(max_chunk[max_silence_index]) != -1:
            chunks.append(max_chunk[:max_silence_index + 1])
            chunks.append(max_chunk[max_silence_index + 1:])
        else:  # NOTE: If ``delimiter`` suggests no cuts, then we ignore ``max_chunk```
            max_chunk_text = script[max_chunk[0].start_text:max_chunk[-1].end_text]
            logger.warning('Unable to cut %s', max_chunk_text)

        max_chunk_index, _ = max(enumerate(chunks), key=lambda a: get_chunk_length(a[1]))
        max_chunk = chunks.pop(max_chunk_index)
        max_chunk_length = get_chunk_length(max_chunk)

    chunks.append(max_chunk)

    # Format the output as text spans and the corresponding audio spans.
    chunks = sorted(chunks, key=lambda c: c[0].start_text)
    spans = []
    for chunk in chunks:
        next_alignment = chunk[-1].next_alignment
        end_text_span = len(script) if next_alignment is None else next_alignment.start_text

        # TODO: Support looking backwards to fetch punctuation like open parentheses and open
        # quotes.
        spans.append({
            'text': (chunk[0].start_text, end_text_span),
            'audio': (chunk[0].start_audio, chunk[-1].end_audio)
        })

    return spans, _review_chunk_alignments(script, chunks, spans)


def setup_io(wav_pattern,
             csv_pattern,
             destination,
             wav_directory_name='wavs',
             csv_metadata_name='metadata.csv',
             gentle_cache_name='.gentle',
             stdout_name='stdout.log',
             stderr_name='stderr.log'):
    """ Perform basic IO operations to setup this script

    Args:
        wav_pattern (str): The audio file glob pattern.
        csv_pattern (str): The CSV file globl pattern containing metadata associated with audio
            file.
        destination (str): Directory to save the processed data.
        wav_directory_name (str, optional): The directory name to store audio clips.
        csv_metadata_name (str, optional): The filename for metadata.
        gentle_cache_name (str, optional): The directory name for Gentle files.
        stdout_name (str): The filename for stdout logs.
        stderr_name (str): The filename for stderr logs.

    Returns:
        wav_paths (list of Path): WAV files to process.
        csv_paths (list of Path): CSV files to process.
        wav_directory (Path): The directory in which to store audio clips.
        gentle_cache_directory (Path): The directory in which to cache gentle requests.
        metadata_filename (Path): The filename to write CSV metadata to.
    """
    destination = Path(destination)
    metadata_filename = destination / csv_metadata_name  # Filename to store CSV metadata
    wav_directory = destination / wav_directory_name  # Directory to store clips
    gentle_cache_directory = destination / gentle_cache_name
    wav_directory.mkdir(parents=True, exist_ok=True)
    gentle_cache_directory.mkdir(exist_ok=True)

    duplicate_stream(sys.stdout, destination / stdout_name)
    duplicate_stream(sys.stderr, destination / stderr_name)

    wav_paths = sorted(list(Path('.').glob(wav_pattern)), key=natural_keys)
    csv_paths = sorted(list(Path('.').glob(csv_pattern)), key=natural_keys)

    assert all(
        [wav_path.suffix == '.wav' for wav_path in wav_paths]), 'Please select only WAV files'
    assert all(
        [csv_path.suffix == '.csv' for csv_path in csv_paths]), 'Please select only CSV files'

    if len(csv_paths) == 0 and len(wav_paths) == 0:
        logger.warning('Found no CSV or WAV files.')
        return
    elif len(csv_paths) != len(wav_paths):
        logger.warning('CSV and WAV files are not aligned.')
        return
    else:
        logger.info('Processing %d files', len(csv_paths))

    return wav_paths, csv_paths, wav_directory, gentle_cache_directory, metadata_filename


def main(wav_pattern,
         csv_pattern,
         destination,
         text_column='Content',
         wav_column='WAV Filename',
         sample_rate=44100,
         max_chunk_length=10):
    """ Align audio with scripts, then create and save chunks of audio and text.

    Args:
        wav_pattern (str): The audio file glob pattern.
        csv_pattern (str): The CSV file globl pattern containing metadata associated with audio
            file.
        destination (str): Directory to save the processed data.
        text_column (str, optional): The script column in the CSV file.
        wav_column (str, optional): Column name for the new audio filename column.
        sample_rate (int, optional): Sample rate of the audio.
        max_chunk_length (float, optional): Number of seconds for the maximum chunk audio length.
    """
    (wav_paths, csv_paths, wav_directory, gentle_cache_directory, metadata_filename) = setup_io(
        wav_pattern, csv_pattern, destination)

    total_unaligned_characters = 0
    total_characters = 0
    for i, (wav_path, csv_path) in enumerate(zip(wav_paths, csv_paths)):
        print('-' * 100)
        script_wav_directory = wav_directory / wav_path.stem
        script_wav_directory.mkdir(exist_ok=True)
        logger.info('Processing %s:%s', wav_path, csv_path)

        logger.info('Reading audio and csv...')
        audio = read_audio(str(wav_path), sample_rate)
        data_frame = pandas.read_csv(csv_path)
        scripts = [x.strip() for x in data_frame[text_column]]

        logger.info('Aligning...')
        try:
            alignments = align_wav_and_scripts(
                wav_path, scripts, gentle_cache_directory, sample_rate=sample_rate)
        except (requests.exceptions.RequestException, ValueError) as e:
            logger.exception('Failed to align %s with %s', wav_path.name, csv_path.name)
            continue

        logger.info('Chunking and writing...')
        to_write = []
        for ((j, row), alignment) in zip(data_frame.iterrows(), alignments):
            chunks, (num_unaligned_characters, len_script) = chunk_alignments(
                alignment, row[text_column], max_chunk_length * sample_rate)
            total_unaligned_characters += num_unaligned_characters
            total_characters += len_script

            for k, chunk in enumerate(chunks):
                new_row = row.to_dict()

                new_row[text_column] = row[text_column][slice(*chunk['text'])].strip()
                audio_filename = script_wav_directory / ('script_%d_chunk_%d.wav' % (j, k))
                new_row[wav_column] = str(audio_filename.relative_to(wav_directory))
                del new_row['Index']  # Delete the default pandas Index column
                to_write.append(new_row)

                audio_span = audio[slice(*chunk['audio'])]
                librosa.output.write_wav(str(audio_filename), audio_span, sr=sample_rate)

        logger.info('Found %d chunks', len(to_write))
        pandas.DataFrame(to_write).to_csv(
            str(metadata_filename), mode='a', header=i == 0, index=False)
    logger.warn('Found %f%% [%d of %d] unaligned characters',
                total_unaligned_characters / total_characters * 100, total_unaligned_characters,
                total_characters)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Align and chunk audio file and text scripts.')
    parser.add_argument('-w', '--wav', type=str, help='Path / Pattern to WAV file to chunk.')
    parser.add_argument('-c', '--csv', type=str, help='Path / Pattern to CSV file with scripts.')
    parser.add_argument('-d', '--destination', type=str, help='Path to save processed files.')
    args = parser.parse_args()
    main(args.wav, args.csv, args.destination)

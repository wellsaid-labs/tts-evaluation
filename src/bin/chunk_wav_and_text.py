import argparse
import logging
import pathlib
import re
import requests

from collections import namedtuple

import pandas
import librosa

from src.audio import read_audio

logging.basicConfig(
    format='[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)

GENTLE_SUCCESS_CASE = 'success'
GENTLE_OOV_WORD = '<unk>'
""" An alignment that was determined confidently between characters and text.

Args:
    start_audio (int): Start of the audio in samples.
    end_audio (int): End of audio in samples.
    start_text (int): Start of text in characters.
    end_text (int): End of text in characters.
    next_start_text (int): Start of next alignement text in characters or end of text.
    next_start_audio (int): Start of next alignement audio in samples or end of audio.
"""
Alignment = namedtuple(
    'Alignment',
    ['start_audio', 'end_audio', 'start_text', 'end_text', 'next_start_text', 'next_start_audio'])


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
    ]), 'Transcript must align with character offsets.'

    # Print warnings
    unaligned_text = None
    last_unaligned_word_index = None
    for i, word in enumerate(response['words']):
        if 'alignedWord' in word:
            continue

        if last_unaligned_word_index and last_unaligned_word_index + 1 == i:
            unaligned_text['endOffset'] = word['endOffset']
            unaligned_text['cases'].add(word['case'])
            last_unaligned_word_index += 1
        else:
            if unaligned_text:
                unaligned_text['text'] = transcript[unaligned_text['startOffset']:unaligned_text[
                    'endOffset']]
                logger.warn('Unaligned text: %s', unaligned_text)

            unaligned_text = {
                'startOffset': word['startOffset'],
                'endOffset': word['endOffset'],
                'cases': set([word['case']])
            }
            last_unaligned_word_index = i

    for word in response['words']:
        if 'alignedWord' not in word or word['alignedWord'] == GENTLE_OOV_WORD:
            continue

        normalized_word = word['word'].lower().replace('’', '\'')
        if word['alignedWord'] != normalized_word:
            logger.warn('``alignedWord`` does not match transcript ``word``: %s', word)

    # Compute statistics
    unaligned_words = sum([w['case'] != GENTLE_SUCCESS_CASE for w in response['words']])
    if unaligned_words > 0:
        logger.warn('%f%% unaligned words', (unaligned_words / len(response['words']) * 100))

    oov_words = [
        w['word']
        for w in response['words']
        if 'alignedWord' in w and w['alignedWord'] == GENTLE_OOV_WORD
    ]
    if len(oov_words) > 0:
        logger.warn('%f%% out of vocabulary words', (len(oov_words) / len(response['words']) * 100))
        logger.warn('Out of vocabulary words: %s', sorted(list(set(oov_words))))


def _request_gentle(wav_path,
                    transcript,
                    hostname='localhost',
                    port=8765,
                    parameters=(('async', 'false'),),
                    wait_per_second_of_audio=0.25,
                    sample_rate=24000):
    """ Align an audio file with the trascript at a word granularity.

    Args:
        wav_path (Path): The audio file path.
        transcript (str): The text spoken in the audio.
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
                    phones (list): This field is not relevant for this repository.
                }, ...
            ]
        }
    """
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

    # TODO: Cache the response on disk
    response = response.json()
    _review_gentle(response, transcript)
    return response


def align_wav_and_scripts(wav_path, scripts, sample_rate):
    """ Align an audio file with the scripts spoken in the audio.

    Notes:
        - The returned alignments do not include the entire script. We only include verified
          and confident alignments.

    Args:
        wav_path (Path): The audio file path.
        scripts (list of str): The scripts spoken in the audio.
        sample_rate (int): Sample rate of the audio.

    Returns:
        (list of Alignment): For every script, a list of found alignments.
    """
    seconds_to_samples = lambda seconds: int(round(seconds * sample_rate))
    max_sample_rate = seconds_to_samples(
        librosa.get_duration(filename=str(wav_path), sr=sample_rate))
    transcript = '\n'.join(scripts)
    alignment = _request_gentle(wav_path, transcript, sample_rate=sample_rate)

    # Align seperate scripts with the transcript alignment.
    aligned_words = [w for w in alignment['words'] if w['case'] == GENTLE_SUCCESS_CASE]
    transcript_start_offset = 0
    all_scripts_alignments = []
    for script in scripts:
        scripts_alignments = []
        transcript_end_offset = transcript_start_offset + len(script)
        while (len(aligned_words) > 0 and aligned_words[0]['endOffset'] <= transcript_end_offset):
            aligned_word = aligned_words.pop(0)

            if len(aligned_words) > 1:
                next_start_text = aligned_words[0]['startOffset'] - transcript_start_offset
                next_start_audio = seconds_to_samples(aligned_words[0]['start'])
            else:
                next_start_text = len(script)
                next_start_audio = max_sample_rate

            scripts_alignments.append(
                Alignment(
                    start_text=aligned_word['startOffset'] - transcript_start_offset,
                    end_text=aligned_word['endOffset'] - transcript_start_offset,
                    start_audio=seconds_to_samples(aligned_word['start']),
                    end_audio=seconds_to_samples(aligned_word['end']),
                    next_start_text=next_start_text,
                    next_start_audio=next_start_audio))

        all_scripts_alignments.append(scripts_alignments)
        transcript_start_offset += len(script) + 1  # Add plus one for ``\n``

    assert len(all_scripts_alignments) == len(
        scripts), 'Alignment was not able to provide alignment for all scripts.'
    assert len(aligned_words) == 0, 'Not all words were aligned with a script.'
    return all_scripts_alignments


def chunk_alignments(alignments, sample_rate, max_chunk_length):
    """ Chunk audio and text to be less than ``max_chunk_length``.

    Notes:
        - We chunk based on the largest silence between alignments. We assume that the returned
          chunks are independent because there exists silence between chunks.
        - Ideal chunk independence means that prior text does not affect the voice actors
          interpretation of that chunk of text.

    Args:
        alignments (list of Alignment): List of alignments between text and audio.
        sample_rate (int): Sample rate of the audio.
        max_chunk_length (float): Number of seconds for the maximum chunk audio length.

    Returns:
        text_spans (list of tuple(int, int)): List of spans of text as measured in characters.
        audio_spans (list of tuple(int, int)): List of spans of audio as measured in samples.
    """
    # Compute the average samples of silence per character
    average_silence = (
        lambda a: (a[1].next_start_audio - a[1].end_audio) / (a[1].next_start_text - a[1].end_text))

    chunks = []
    largest_chunk = alignments
    largest_chunk_length = largest_chunk[-1].next_start_audio - largest_chunk[0].start_audio
    while len(largest_chunk) > 1 and largest_chunk_length > max_chunk_length * sample_rate:
        # TODO: Keep a list of the largest silences instead of recomputing every iteration
        # NOTE: Computes the largest average silence per character.
        largest_silence_index, _ = max(enumerate(largest_chunk[:-1]), key=average_silence)

        chunks.append(largest_chunk[:largest_silence_index + 1])
        chunks.append(largest_chunk[largest_silence_index + 1:])

        # TODO: Keep a list of the largest chunks instead of recomputing every iteration
        largest_chunk_index, _ = max(
            enumerate(chunks), key=lambda c: c[1][-1].next_start_audio - c[1][0].start_audio)
        largest_chunk = chunks.pop(largest_chunk_index)
        largest_chunk_length = largest_chunk[-1].next_start_audio - largest_chunk[0].start_audio

    chunks.append(largest_chunk)
    chunks = sorted(chunks, key=lambda c: c[0].start_text)
    text_spans = []
    audio_spans = []
    for i, chunk in enumerate(chunks):
        # NOTE: Given that alignments do not cover all the text, we extend the spans to the next
        # start.
        # TODO: Should we cut the audio up to next_start_audio, end_audio or in between? If there
        # are unaligned words then next_start_audio makes sense. If next_start_audio isn't
        # quite accurate then in between makes sense.
        # TODO: Consider not expanding the text passed the end_text except in cases of punctutation;
        # look at other cases where alignment does not occur.
        text_spans.append((chunk[0].start_text, chunk[-1].next_start_text))
        audio_spans.append((chunk[0].start_audio, chunk[-1].end_audio))

    return text_spans, audio_spans


def main(wav_pattern,
         csv_pattern,
         text_column='Content',
         wav_column='WAV Filename',
         destination='data/processed',
         wav_directory_name='wav',
         csv_metadata_name='metadata.csv',
         sample_rate=44100,
         max_chunk_length=10):
    """ Align audio with scripts, then create and save chunks of audio and text.

    Args:
        wav_pattern (str): The audio file glob pattern.
        csv_pattern (str): The CSV file globl pattern containing metadata associated with audio
            file.
        text_column (str, optional): The script column in the CSV file.
        wav_column (str, optional): Column name for the new audio filename column.
        wav_directory_name (str, optional): The directory name to store audio clips.
        csv_metadata_name (str, optional): The filename to store metadata.
        sample_rate (int, optional): Sample rate of the audio.
        max_chunk_length (float, optional): Number of seconds for the maximum chunk audio length.
    """
    root_directory = pathlib.Path(destination)
    root_directory.mkdir(parents=True, exist_ok=True)
    metadata_filename = root_directory / csv_metadata_name  # Filename to store CSV metadata
    wav_directory = root_directory / wav_directory_name  # Directory to store clips
    wav_directory.mkdir(parents=True, exist_ok=True)

    wav_paths = sorted(list(pathlib.Path('.').glob(wav_pattern)), key=natural_keys)
    csv_paths = sorted(list(pathlib.Path('.').glob(csv_pattern)), key=natural_keys)
    for i, (wav_path, csv_path) in enumerate(zip(wav_paths, csv_paths)):
        is_first_script = i == 0
        if not is_first_script:
            print('-' * 100)

        script_wav_directory = (wav_directory / ('script_%d' % i))
        if script_wav_directory.is_dir():
            logger.info('Skipping cached files %s:%s', wav_path, csv_path)
            continue

        logger.info('Processing %s:%s', wav_path, csv_path)

        logger.info('Reading audio...')
        audio = read_audio(str(wav_path), sample_rate)

        logger.info('Reading csv...')
        df = pandas.read_csv(csv_path)
        scripts = [x.strip() for x in df[text_column]]

        logger.info('Aligning...')
        try:
            alignments = align_wav_and_scripts(wav_path, scripts, sample_rate=sample_rate)
        except (requests.exceptions.RequestException, ValueError) as e:
            logger.exception('Failed to align %s with %s', wav_path.name, csv_path.name)
            continue

        script_wav_directory.mkdir()
        logger.info('Chunking and writing...')
        to_write = []
        for j, row in df.iterrows():
            if len(alignments[j]) == 0:
                logger.warn('Unable to align script %d', j)
                continue

            text_spans, audio_spans = chunk_alignments(
                alignments[j], sample_rate=sample_rate, max_chunk_length=max_chunk_length)

            for k, (text_span, audio_span) in enumerate(zip(text_spans, audio_spans)):
                new_row = row.to_dict()
                new_row[text_column] = row[text_column][slice(*text_span)].strip()
                new_row[wav_column] = str(
                    script_wav_directory / ('script_%d_chunk_%d.wav' % (j, k)))
                to_write.append(new_row)
                audio_span = audio[slice(*audio_span)]
                librosa.output.write_wav(new_row[wav_column], audio_span, sr=sample_rate)

        logger.info('Found %d chunks', len(to_write))
        pandas.DataFrame(to_write).to_csv(
            str(metadata_filename), mode='a', header=is_first_script, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Align and chunk audio file and text scripts.')
    parser.add_argument('-w', '--wav', type=str, help='Path / Pattern to WAV file to chunk.')
    parser.add_argument('-c', '--csv', type=str, help='Path / Pattern to CSV file with scripts.')
    args = parser.parse_args()
    main(args.wav, args.csv)

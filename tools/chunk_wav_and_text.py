import argparse
import logging
import pathlib
import requests

from functools import partial

import pandas
import librosa

from src.audio import read_audio

logger = logging.getLogger(__name__)


def _request_gentle(wav_path,
                    transcript,
                    hostname='localhost',
                    port=8765,
                    parameters=(('async', 'false'),)):
    """ Align an audio file with the trascript at a word granularity.

    Args:
        wav_path (Path): The audio file path.
        transcript (str): The text spoken in the audio.
        hostname (str, optional): Hostname used by the gentle server.
        port (int, optional): Port used by the gentle server.
        parameters (tuple, optional): Dictionary or bytes to be sent in the query string for the
            :class:`Request`.

    Returns:
        (dict) {
            transcript (str): The transcript passed in.
            words (list) [
                {
                    alignedWord (str): Normalized and aligned word.
                    word (str): Parallel unnormalized word from transcript.
                    case (str): One of three cases 'success', 'not-found-in-audio', and
                        'not-found-in-transcript'.
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
    url = 'http://{}:{}/transcriptions'.format(hostname, port)
    response = requests.post(
        url,
        params=parameters,
        files={
            'audio': wav_path.read_bytes(),
            'transcript': transcript.encode(),
        })
    if response.status_code != 200:
        raise ValueError('Gentle ({}) returned bad response: {}'.format(url, response.status_code))
    response = response.json()

    # Check various invariants
    assert response['transcript'] == transcript, 'Failed transcript invariant.'
    assert all([
        transcript[w['startOffset']:w['endOffset']] == w['word'] for w in response['words']
    ]), 'Transcript must align with character offsets.'
    for word in response['words']:
        if 'alignedWord' not in word:
            logger.warn('``alignedWord`` does not exist — %s', word)
        elif word['alignedWord'] != word['word'].lower():
            logger.warn('``alignedWord`` does not match trascript ``word`` — %s', word)
    unaligned_words = sum([w['case'] != 'success' for w in response['words']])
    if unaligned_words > 0:
        logger.warn('%f%% unaligned words', (unaligned_words / len(response['words']) * 100))

    return response


def align_wav_and_scripts(wav_path, scripts, sample_rate=44100):
    """ Align an audio file with the scripts spoken in the audio.

    Args:
        wav_path (Path): The audio file path.
        scripts (list of str): The scripts spoken in the audio.
        sample_rate (int, optional)

    Returns:
        (list) [
            (list) [  # For every script, a list of words and their alignment.
               {
                    alignedWord (str): Normalized and aligned word.
                    word (str): Parallel unnormalized word from transcript.
                    case (str): One of three cases 'success', 'not-found-in-audio', and
                        'not-found-in-transcript'.
                    start (float): Time the speakers begins to speak ``word`` in samples.
                    end (float): Time the speakers stops speaking ``word`` in samples.
                    startOffset (int): The script character index of the first letter of word.
                    endOffset (int): The script character index plus one of the last letter of word.
                }, ...
            ]
        ]
    """
    transcript = '\n'.join(scripts)
    alignment = _request_gentle(wav_path, transcript)

    # Align seperate scripts with the transcript alignment.
    aligned_words = alignment['words']
    transcript_start_offset = 0
    scripts_alignment = []
    for script in scripts:
        script_words = []
        while (len(aligned_words) > 0 and
               aligned_words[0]['endOffset'] <= transcript_start_offset + len(script)):
            aligned_word = aligned_words.pop(0)
            aligned_word['startOffset'] -= transcript_start_offset

            # Document this change
            if len(aligned_words) > 1:
                aligned_word[
                    'endOffset'] = aligned_words[0]['startOffset'] - transcript_start_offset - 1
                assert aligned_word['endOffset'] > aligned_word['startOffset']
            else:
                aligned_word['endOffset'] = len(script)
                assert aligned_word['endOffset'] > aligned_word['startOffset']

            if aligned_word['case'] == 'success':
                del aligned_word['phones']
                aligned_word['start'] = int(round(aligned_word['start'] * sample_rate))
                aligned_word['end'] = int(round(aligned_word['end'] * sample_rate))
            script_words.append(aligned_word)
        scripts_alignment.append(script_words)

        transcript_start_offset += len(script) + 1  # Add plus one for ``\n``
    assert len(scripts_alignment) == len(
        scripts), 'Alignment was not able to provide alignment for all scripts.'
    return scripts_alignment


def compute_silence(sequence, index):
    return (sequence[index - 1]['end'] - sequence[index]['start']) / (
        sequence[index - 1]['endOffset'] - sequence[index]['startOffset'])


def main(wav_path, csv_path, text_column='Content', max_clip_length=10):
    """
    Args:
        wav_path (Path): The audio file path.
        csv_path (Path): The CSV file path containing metadata associated with audio file.
        text_column (str, optional): The script column in the CSV file.
        max_clip_length (float, optional): max clip length in seconds.
    """
    assert wav_path.is_file(), 'Audio file must exist'
    assert csv_path.is_file(), 'CSV file must exist'

    audio = read_audio(str(wav_path), 44100)
    df = pandas.read_csv(csv_path)
    scripts = [x.strip() for x in df[text_column]]
    df['alignment'] = align_wav_and_scripts(wav_path, scripts)

    text_directory = pathlib.Path('data/text')
    text_directory.mkdir(parents=True, exist_ok=True)

    wav_directory = pathlib.Path('data/wav')
    wav_directory.mkdir(parents=True, exist_ok=True)

    for i, row in df.iterrows():
        chunks = []
        max_chunk = [word for word in row['alignment'] if word['case'] == 'success']
        max_chunk_length = max_chunk[-1]['end'] - max_chunk[0]['start']
        while len(max_chunk) > 1 and max_chunk_length > max_clip_length * 44100:
            # TODO: Do something about the fact that not all have a start_end but that doesn't
            # mean there is a large silence there...
            max_silence_index = max(
                range(1, len(max_chunk)), key=partial(compute_silence, max_chunk))

            chunks.append(max_chunk[:max_silence_index])
            chunks.append(max_chunk[max_silence_index:])

            max_chunk_index = max(
                range(len(chunks)), key=lambda i: chunks[i][-1]['end'] - chunks[i][0]['start'])
            max_chunk = chunks.pop(max_chunk_index)
            max_chunk_length = max_chunk[-1]['end'] - max_chunk[0]['start']

        chunks.append(max_chunk)

        for j, chunk in enumerate(chunks):
            # TODO: endOffset should be -1 of the next word in case anything like periods were
            # missed
            script = row[text_column]
            text_chunk = script[chunk[0]['startOffset']:min(chunk[-1]['endOffset'] +
                                                            1, len(script))]
            assert chunk[-1]['end'] <= audio.shape[0]
            audio_chunk = audio[chunk[0]['start']:chunk[-1]['end']]
            librosa.output.write_wav(
                str(wav_directory / ('%d_%d.wav' % (i, j))), audio_chunk, sr=44100)
            (text_directory / ('%d_%d.txt' % (i, j))).write_text(text_chunk)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Align and chunk audio file and text scripts.')
    parser.add_argument('-w', '--wav', type=str, help='Path to WAV file to chunk.')
    parser.add_argument('-c', '--csv', type=str, help='Path to CSV file with scripts.')
    args = parser.parse_args()
    main(pathlib.Path(args.wav), pathlib.Path(args.csv))

import argparse
import pathlib
import requests

import pandas


def align_wav_and_text(wav_path,
                       transcript,
                       hostname='localhost',
                       port=8765,
                       parameters=(('async', 'false'),)):
    """
    Args:
        wav_path (Path)
        transcript (str)
        hostname (str)
        port (int)
        parameters (tuple)

    Returns:
        (dict) {
            transcript (str): The transcript passed in.
            words (list) [
                {
                    alignedWord (str): Normalized and aligned word.
                    word (str): Parallel unnormalized word from transcript.
                    case (str): One of three cases 'success', 'not-found-in-audio', and
                        'not-found-in-transcript'.
                    start (float): Time the speakers begins to speak ``word``.
                    end (float): Time the speakers stops speaking ``word``.
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
    assert all(
        [transcript[w['startOffset']:w['endOffset']] == w['word'] for w in response['words']])
    unaligned_words = sum([w['case'] != 'success' for w in response['words']])
    print('%f unaligned words' % unaligned_words / len(response['words']))
    return response


def main(wav_path, csv_path, text_column='Content'):
    """
    Args:
        wav_path (Path)
        csv_path (Path)
        text_column (str)
    """
    assert wav_path.is_file(), "Audio file must exist"
    assert csv_path.is_file(), "CSV file must exist"

    df = pandas.read_csv(csv_path)
    scripts = [x.strip() for x in df[text_column]]
    transcript = '\n'.join(scripts)
    print(align_wav_and_text(wav_path, transcript))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Align and chunk audio file and text scripts.')
    parser.add_argument('-w', '--wav', type=str, help='Path to WAV file to chunk.')
    parser.add_argument('-c', '--csv', type=str, help='Path to CSV file with scripts.')
    args = parser.parse_args()
    main(pathlib.Path(args.wav), pathlib.Path(args.csv))

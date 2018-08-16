#!/usr/bin/python3

import json
import argparse
from pydub import AudioSegment
import pandas
from pandas import DataFrame
import numpy as np
import sys


NO_ACTION = False

def get_line_spacing(script_csv_path: str, transcript: str):
    """Given a script CSV file and a transcript, return a DataFrame with the
       byte offsets for each script entry into the transcript.
    """
    # Load the data off the filesystem
    script_df = get_script(script_csv_path)

    # Create a np array long enough to hold one entry for each script line
    spacing = np.empty(len(script_df))

    # Record the byte-wise location of each line, as that is how gentle reports
    # word locations
    entry_offset = 0
    for entry in script_df.iterrows():
        entry_offset = transcript.find(entry[1].Content.strip(), entry_offset)
        print('Found ' + str(entry[1].Index) + ' at character ' + str(entry_offset))
        if entry_offset == -1:
            print(' Unable to find string: [' + str(entry[1].Content) + ']')
        spacing[entry[1].Index] = entry_offset

    return spacing

def find_matching_line(spacing: np.ndarray, startOffset: int):
    return np.argmin(spacing <= startOffset) - 1

def get_timing_data(spacing: np.ndarray, gentle_words: dict):
    """Iterate over each word identified by gentle and record the lowest and
       highest timing information for the script line identified by the
       startOffset character.
    """
    # Create a timing array to store the earliest and latest times identified
    timing = np.empty([spacing.shape[0], 2])
    for i in range(spacing.shape[0]):
        timing[i][0] = sys.maxsize          # Some overly large number
        timing[i][1] = 0

    # For each word, check to see if it's the first (or last) word for a given
    # script line.
    for word in gentle_words:
        line = find_matching_line(spacing, word['startOffset'])
        try:
            timing[line][0] = min(timing[line][0], word['start'] * 1000)
            timing[line][1] = max(timing[line][1], word['end'] * 1000)
            # print(line, word['startOffset'], word['alignedWord'], word['start'], word['end'])
        except KeyError:
            # Entries that aren't matched to speach aren't fully populated
            pass

    return timing

def get_script_timing(script_csv_path: str, gentle_path: str):
    """Identifies the correct location (in ms) for each line in the supplied
       script, as specified by the results from the gentle request.

       Returns a array of [start_ms, end_ms] tuples, one per line in the script.
    """
    with open(gentle_path, 'r') as gentle_file:
        gentle_result = json.load(gentle_file)
    
    # Corrolate the transcript's byte-locations with the script lines
    spacing = get_line_spacing(script_csv_path, gentle_result['transcript'])

    # Get the timing information for each line
    timing = get_timing_data(spacing, gentle_result['words'])

    return timing

def get_script(script_csv_path: str):
    try:
        script_df = pandas.read_csv(script_csv_path)
    except pandas.errors.ParserError:
        script_df = pandas.read_csv(script_csv_path, sep='\t')
    return script_df

def split_wav(wav_path: str, timing: np.ndarray, prefix: str, tags: dict, script_csv_path: str):
    wav = AudioSegment.from_file(wav_path, format='wav')
    script_df = get_script(script_csv_path)

    for line_id in range(timing.shape[0]):
        start_ms = timing[line_id][0]
        end_ms = timing[line_id][1]

        print(f'{line_id} - {start_ms:.0f} -> {end_ms:.0f}')

        if NO_ACTION:
            continue

        file_tags = tags.copy()
        file_tags['line_id'] = line_id
        file_tags['line'] = script_df.iloc(0)[line_id].Content
        file_tags['start'] = start_ms
        file_tags['end'] = end_ms

        sample = wav[start_ms : end_ms]
        with open(prefix + str(line_id) + '.wav', 'wb') as f, open(prefix + str(line_id) + '.json', 'w') as j:
            sample.export(f, format='wav')
            j.write(json.dumps(file_tags, indent=2))

def main(wav_path: str, script_csv_path: str, gentle_path: str, dest: str, tags: dict):
    """"""
    timing = get_script_timing(script_csv_path, gentle_path)

    split_wav(wav_path, timing, dest, tags, script_csv_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Split a wav file into word tokens')
    parser.add_argument("-n",
                        help="Output expected behavior only, no action taken.",
                        default=False, action='store_true')
    parser.add_argument("wav", type=str,
                        help="Wav file input.")
    parser.add_argument("script", type=str,
                        help="Script CSV file input.")
    parser.add_argument("gentle", type=str,
                        help="Gentle response json file.")
    parser.add_argument("dest", type=str,
                        help="Directory to record output snippets.")
    parser.add_argument("tags", type=str,
                        help="Tags, in JSON format",
                        default='{}')
    args = parser.parse_args()

    print('Tags', args.tags)
    if args.n:
        NO_ACTION = True

    main(args.wav, args.script, args.gentle, args.dest, json.loads(args.tags))


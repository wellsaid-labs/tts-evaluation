#!/usr/bin/python3

import argparse
import pandas
import csv

def main(path: str):
    """Strip the unneeded columns from a script csv file."""
    try:
        df = pandas.read_csv(path)
    except FileNotFoundError:
        print(f'Error: \'{path}\' was not found.')
        exit(-1)
    contents = [x.strip() for x in df['Content']]
    print('\n'.join(contents))

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Strip unnecessary columns out of the supplied script')
    parser.add_argument("path", type=str,
                        help="Path to the CSV file to parse.")
    args = parser.parse_args()

    main(args.path)


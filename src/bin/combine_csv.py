""" Combine CSVs into a combined CSV file.

Example:

    python src/bin/combine_csv.py --csvs 'Script 52.csv' 'Script 53.csv' 'Script 54.csv' \
                                   --name 'Scripts 52-54.csv'

"""
import argparse
import pandas

import numpy as np


def main(csvs, name, shuffle=False):
    """
    NOTE: This module also adds an additional "__csv" column to preserve the row origin.

    Args:
        csvs (list of str): List of CSV filenames.
        name (str): Output filename.
    """
    df = pandas.read_csv(csvs[0])
    df['__csv'] = csvs[0]
    for csv in csvs[1:]:
        df_csv = pandas.read_csv(csv)
        df_csv['__csv'] = csv
        df = df.append(df_csv, ignore_index=True)
    if shuffle:
        df = df.iloc[np.random.permutation(len(df))]
    df.to_csv(name, index=False)


if __name__ == '__main__':  # pragma: no cover
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--csvs', default=[], required=True, nargs='+', help='List of CSV documents to combine.')
    parser.add_argument(
        '--shuffle', action='store_true', default=False, help='Shuffle the CSV rows randomly.')
    parser.add_argument('--name', type=str, required=True, help='Name of the output file.')
    args = parser.parse_args()

    main(args.csvs, args.name, args.shuffle)

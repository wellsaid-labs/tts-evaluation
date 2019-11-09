""" Add a prefix to a CSV column.

Example:

    python src/bin/combine_csv.py --csv metadata.csv \
        --column_name audio_path \
        --prefix https://storage.googleapis.com/mturk-samples/2019-08-20/ \
        --destination metadata.csv

"""
import argparse
import pandas


def main(csv, column_name, prefix, destination):
    """
    Add a prefix to `column_name` in `csv` and then save to `destination`.

    Args:
        csv (str): The CSV file to load
        column_name (str): The column name to adjust.
        prefix (str): The prefix to add.
        destination (str): The new CSV file.
    """
    df = pandas.read_csv(csv)
    df[column_name].add_prefix('item_')
    df.to_csv(destination)


if __name__ == '__main__':  # pragma: no cover
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, required=True, help='The CSV file to load.')
    parser.add_argument('--column_name', type=str, required=True, help='The column name to adjust.')
    parser.add_argument('--prefix', type=str, required=True, help='The prefix to add.')
    parser.add_argument('--destination', type=str, required=True, help='The new CSV file.')
    args = parser.parse_args()

    main(args.csv, args.column_name, args.prefix, args.destination)

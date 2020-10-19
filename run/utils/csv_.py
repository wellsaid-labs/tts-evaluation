""" Command-line interface (CLI) for processing CSV files. """
import math
import pathlib
import typing

import numpy as np
import pandas
import typer

app = typer.Typer(context_settings=dict(max_content_width=math.inf))


@app.command()
def combine(
    csvs: typing.List[pathlib.Path] = typer.Argument(..., help="List of CSVs to combine."),
    csv: pathlib.Path = typer.Argument(..., help="Combined CSV filename."),
):
    """Combine a list of CSVS input one CSV.

    Also, this adds an additional '__csv' column with the original filename.
    """
    df = pandas.read_csv(csvs[0])
    df["__csv"] = csvs[0]
    for csv in csvs[1:]:
        df_csv = pandas.read_csv(csv)
        df_csv["__csv"] = csv
        df = df.append(df_csv, ignore_index=True)
    df.to_csv(csv, index=False)


@app.command()
def shuffle(source: pathlib.Path, destination: pathlib.Path):
    """ Shuffle SOURCE csv and save it at DESTINATION. """
    df = pandas.read_csv(source)
    df = df.iloc[np.random.permutation(len(df))]
    df.to_csv(destination, index=False)


@app.command()
def prefix(source: pathlib.Path, destination: pathlib.Path, column: str, prefix: str):
    """ Add a PREFIX to every entry in the SOURCE csv under COLUMN and save to DESTINATION. """
    df = pandas.read_csv(source)
    df[column] = prefix + df[column].astype(str)
    df.to_csv(destination, index=False)


if __name__ == "__main__":
    app()

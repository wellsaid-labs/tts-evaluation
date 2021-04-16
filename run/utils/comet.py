""" Command-line interface (CLI) for Comet. """
import math
import pathlib
import subprocess

import comet_ml
import tqdm
import typer

import lib
import run

app = typer.Typer(context_settings=dict(max_content_width=math.inf))


@app.command()
def checkout(experiment: str):
    """ Updates the files in the working directory to reproduce a EXPERIMENT. """
    api = comet_ml.api.API()
    experiment_ = api.get_experiment_by_id(experiment)
    git = experiment_.get_git_metadata()
    typer.secho(f"Checking out experiment {experiment_.get_name()}...", fg=typer.colors.MAGENTA)
    subprocess.run(f"git checkout {git['branch']}", shell=True, check=True)
    subprocess.run(f"git checkout {git['parent']}", shell=True, check=True)
    patch.__wrapped__(experiment)  # type: ignore


@app.command()
def patch(experiment: str):
    """ Apply git patch created for a EXPERIMENT. """
    api = comet_ml.api.API()
    experiment_ = api.get_experiment_by_id(experiment)
    patch = experiment_.get_git_patch().decode()
    subprocess.run(f"echo {patch} | git apply")


@app.command()
def samples(
    experiment: str,
    dest: pathlib.Path = run._config.SAMPLES_PATH / lib.environment.bash_time_label(),
):
    """ Download all samples for an EXPERIMENT. """
    api = comet_ml.api.API()
    experiment_ = api.get_experiment_by_id(experiment)
    asset_list = experiment_.get_asset_list(asset_type="audio")
    dest.mkdir()
    typer.echo(f"Downloading samples for experiment {experiment} into {dest}...")
    for asset in tqdm.tqdm(asset_list):
        file_path = dest / str(asset["fileName"]).replace(experiment, "")
        file_path.write_bytes(experiment_.get_asset(asset["assetId"], return_type="binary"))


if __name__ == "__main__":
    app()

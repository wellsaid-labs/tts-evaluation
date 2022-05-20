""" Command-line interface (CLI) for Comet. """
import math
import pathlib
import subprocess
import typing
import zipfile

import comet_ml
import tqdm
import typer

import lib
import run

app = typer.Typer(context_settings=dict(max_content_width=math.inf))


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def checkout(context: typer.Context, experiment: str, overwrite: bool = False):
    """Updates the files in the working directory to reproduce a EXPERIMENT."""
    api = comet_ml.api.API()
    experiment_ = api.get_experiment_by_key(experiment)
    assert experiment_ is not None
    git = experiment_.get_git_metadata()
    typer.secho(f"Checking out experiment {experiment_.get_name()}...", fg=typer.colors.MAGENTA)
    subprocess.run(f"git checkout {git['branch']}", shell=True, check=True)
    subprocess.run(f"git checkout {git['parent']}", shell=True, check=True)
    patch.__wrapped__(context, experiment, overwrite=overwrite)  # type: ignore


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def patch(
    context: typer.Context,
    experiment: str,
    zip_file_name: str = "patch.zip",
    overwrite: bool = False,
):
    """Apply git patch created for a EXPERIMENT.

    Examples:

    $ python -m run.utils.comet patch XXXXXXXX --overwrite --include="*.py"

    $ python -m run.utils.comet patch XXXXXXXX --overwrite --ignore-space-change \
--ignore-whitespace --3way --include="*.py"

    $ python -m run.utils.comet patch XXXXXXXX --overwrite --ignore-space-change \
--ignore-whitespace --reject --include="*.py"
    """
    api = comet_ml.api.API()
    patch_file_name = "git_diff.patch"
    experiment_ = api.get_experiment_by_key(experiment)
    assert experiment_ is not None
    patch = typing.cast(bytes, experiment_.get_git_patch())
    zip_file_path = pathlib.Path(zip_file_name)
    if zip_file_path.exists() and not overwrite:
        typer.echo(f"File {zip_file_path} already exists.")
        raise typer.Exit()
    zip_file_path.write_bytes(patch)
    with zipfile.ZipFile(zip_file_name) as zip_:
        with zip_.open(patch_file_name) as patch:
            patch_file_path = pathlib.Path(patch_file_name)
            if patch_file_path.exists() and not overwrite:
                typer.echo(f"File {patch_file_path} already exists.")
                raise typer.Exit()
            patch_file_path.write_bytes(patch.read())
    subprocess.run(["git", "apply", patch_file_name] + context.args)


@app.command()
def samples(
    experiment: str,
    dest: pathlib.Path = run._config.SAMPLES_PATH / lib.environment.bash_time_label(),
    max_samples: int = 100,
):
    """Download all samples for an EXPERIMENT."""
    api = comet_ml.api.API()
    experiment_ = api.get_experiment_by_key(experiment)
    assert experiment_ is not None
    asset_list = experiment_.get_asset_list(asset_type="audio")
    asset_list = typing.cast(typing.List, asset_list)
    asset_list = sorted(asset_list, key=lambda a: a["createdAt"], reverse=True)[:max_samples]
    dest.mkdir()
    typer.echo(f"Downloading samples for experiment {experiment} into {dest}...")
    for asset in tqdm.tqdm(asset_list):
        file_path = dest / str(asset["fileName"]).replace(experiment, "")
        file_path.write_bytes(experiment_.get_asset(asset["assetId"], return_type="binary"))


if __name__ == "__main__":
    app()

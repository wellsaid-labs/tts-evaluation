""" Command-line interface (CLI) for Comet. """
import math
import subprocess

import comet_ml
import typer

app = typer.Typer(context_settings=dict(max_content_width=math.inf))


@app.command()
def checkout(experiment: str):
    """ Updates the files in the working directory to reproduce a EXPERIMENT. """
    api = comet_ml.api.API()
    experiment_ = api.get_experiment_by_id(experiment)
    git = experiment_.get_git_metadata()
    typer.secho(
        f"Checking out experiment {experiment_.get_name()}...", fg=typer.colors.MAGENTA
    )
    subprocess.run(f"git checkout {git['branch']}", shell=True, check=True)
    subprocess.run(f"git checkout {git['parent']}", shell=True, check=True)
    patch.__wrapped__(experiment)  # type: ignore


@app.command()
def patch(experiment: str):
    """ Apply git patch created for a EXPERIMENT. """
    api = comet_ml.api.API()
    experiment_ = api.get_experiment_by_id(experiment)
    patch = experiment_.get_git_patch()
    subprocess.run(f"echo {patch} | git apply")


if __name__ == "__main__":
    app()

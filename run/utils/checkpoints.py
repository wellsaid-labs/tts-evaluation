""" Command-line interface (CLI) for handling checkpoints. """
import math
import pathlib
import subprocess
from concurrent import futures

import typer

from lib.environment import PT_EXTENSION, ROOT_PATH
from run._config.environment import EXPERIMENTS_PATH, REMOTE_ROOT_PATH
from run.utils.gcp.__main__ import get_running_instances

app = typer.Typer(context_settings=dict(max_content_width=math.inf))

REMOTE_EXPERIMENTS_PATH = f"/{REMOTE_ROOT_PATH}/{EXPERIMENTS_PATH.relative_to(ROOT_PATH)}"


def get_all_checkpoints(vm_zone: str, vm_name: str, dir_name: str) -> bytes:
    """Get all experimental checkpoints on VM for the directory `dir_name`.

    Example:
    >>> get_all_checkpoints("us-east1-c", "michaelpetrochuk-experiment-1-kmn7", "spectrogram_model")
    2023-04-17+12:33:41.1057933780 DATE-2023y04m12d-05h34m45s_PID-29454/.../step_495008.pt
    2023-04-12+07:00:20.8304071980 DATE-2023y04m12d-05h34m45s_PID-29454/.../step_0.pt
    2023-04-12+08:43:46.2595035990 DATE-2023y04m12d-05h34m45s_PID-29454/.../step_7984.pt
    2023-04-12+07:58:55.8837337930 DATE-2023y04m12d-05h34m45s_PID-29454/.../step_3992.pt
    2023-04-16+13:00:27.9580116800 DATE-2023y04m12d-05h34m45s_PID-29454/.../step_391216.pt
    2023-04-16+12:01:07.3030038620 DATE-2023y04m12d-05h34m45s_PID-29454/.../step_387224.pt
    """
    subcommand = f"cd {REMOTE_EXPERIMENTS_PATH}/{dir_name}/; "
    subcommand += f"find . -name '*{PT_EXTENSION}' -mindepth 1 -printf '%T+ %P\\n'"
    command = f'gcloud compute ssh --zone={vm_zone} {vm_name} --command="{subcommand}"'
    return subprocess.check_output(command, shell=True)


def get_latest_checkpoint(vm_zone: str, vm_name: str, dir_name: str) -> str:
    """Get all experimental checkpoints on VM for the directory `dir_name`.

    Example:
    >>> get_latest_checkpoint("us-east1-c", "michaelpetrochuk-experiment-1-kmn7",
    "spectrogram_model")
    DATE-2023y04m12d-05h34m45s_PID-29454/RUN_DATE-2023y04m17d-12h31m01s/checkpoints/step_495008.pt
    """
    checkpoints = get_all_checkpoints(vm_zone, vm_name, dir_name).decode("utf-8")
    checkpoints = sorted(checkpoints.split("\n"))
    return checkpoints[-1].split(" ")[-1]


def download_checkpoint(
    vm_zone: str, vm_name: str, dir_name: str, checkpoint_path: pathlib.Path
) -> pathlib.Path:
    """Download an experimental checkpoints from VM in the directory `dir_name`
    at `checkpoint_path`."""
    destination = EXPERIMENTS_PATH / dir_name / vm_name
    destination.mkdir(exist_ok=True)
    source = f"{REMOTE_EXPERIMENTS_PATH}/{dir_name}/{checkpoint_path}"
    command = f"gcloud compute scp {vm_name}:{source} {destination} --zone={vm_zone}"
    subprocess.run(command, shell=True)
    return destination / checkpoint_path.name


@app.command(help="Get all checkpoints.")
def all(vm_zone: str, vm_name: str, dir_name: str):
    typer.echo(get_all_checkpoints(vm_zone, vm_name, dir_name), nl=False)


@app.command(help="Get the latest checkpoint.")
def latest(vm_zone: str, vm_name: str, dir_name: str):
    typer.echo(get_latest_checkpoint(vm_zone, vm_name, dir_name), nl=False)


@app.command(help="Download a checkpoint.")
def download(vm_zone: str, vm_name: str, dir_name: str, path: pathlib.Path):
    download_checkpoint(vm_zone, vm_name, dir_name, path)


@app.command(help="Download the latest checkpoint.")
def download_latest(vm_zone: str, vm_name: str, dir_name: str):
    latest = pathlib.Path(get_latest_checkpoint(vm_zone, vm_name, dir_name))
    download_checkpoint(vm_zone, vm_name, dir_name, latest)


@app.command(help="Download the latest checkpoint from all running instances.")
def download_all_latest(prefix: str, dir_name: str):
    instances = get_running_instances(prefix)
    with futures.ThreadPoolExecutor(max_workers=len(instances)) as pool:
        list(pool.map(lambda i: download_latest(i[1], i[0], dir_name), instances))


if __name__ == "__main__":
    app()

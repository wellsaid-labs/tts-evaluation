import math

import typer

from lib.environment import ROOT_PATH, save
from run._config import TTS_PACKAGE_PATH
from run._tts import CHECKPOINTS_LOADERS, Checkpoints, package_tts

app = typer.Typer(context_settings=dict(max_content_width=math.inf))


def main(checkpoints: Checkpoints, overwrite: bool = False):
    """Create a TTS package for running inference with CHECKPOINTS."""
    package = package_tts(*CHECKPOINTS_LOADERS[checkpoints]())
    save(TTS_PACKAGE_PATH, package, overwrite=overwrite)
    typer.echo(TTS_PACKAGE_PATH.relative_to(ROOT_PATH))


if __name__ == "__main__":  # pragma: no cover
    typer.run(main)

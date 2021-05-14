import math
import pathlib
from typing import cast

import typer

from lib.environment import load, save
from run import train
from run._config import TTS_BUNDLE_PATH
from run._tts import make_tts_bundle

app = typer.Typer(context_settings=dict(max_content_width=math.inf))


def main(
    spectrogram_checkpoint: pathlib.Path,
    signal_checkpoint: pathlib.Path,
    overwrite: bool = False,
):
    """Make a bundle of inference components from the SPECTROGRAM_CHECKPOINT and the
    SIGNAL_CHECKPOINT."""
    spec_ckpt = cast(train.spectrogram_model._worker.Checkpoint, load(spectrogram_checkpoint))
    sig_ckpt = cast(train.signal_model._worker.Checkpoint, load(signal_checkpoint))
    bundle = make_tts_bundle(spec_ckpt, sig_ckpt)
    save(TTS_BUNDLE_PATH, bundle, overwrite=overwrite)
    typer.echo(TTS_BUNDLE_PATH)


if __name__ == "__main__":  # pragma: no cover
    typer.run(main)

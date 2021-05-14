import math
import pathlib

import typer

from lib.environment import ROOT_PATH, load, save
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
    spec_ckpt = load(spectrogram_checkpoint)
    sig_ckpt = load(signal_checkpoint)
    assert isinstance(spec_ckpt, train.spectrogram_model._worker.Checkpoint)
    assert isinstance(sig_ckpt, train.signal_model._worker.Checkpoint)
    bundle = make_tts_bundle(spec_ckpt, sig_ckpt)
    save(TTS_BUNDLE_PATH, bundle, overwrite=overwrite)
    typer.echo(TTS_BUNDLE_PATH.relative_to(ROOT_PATH))


if __name__ == "__main__":  # pragma: no cover
    typer.run(main)

import dataclasses
import logging
import math
import pathlib
from typing import cast

import typer

from lib.environment import load, save
from lib.signal_model import SignalModel
from lib.spectrogram_model import SpectrogramModel
from run import train
from run._config import TTS_BUNDLE_PATH

app = typer.Typer(context_settings=dict(max_content_width=math.inf))
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class TTSBundle:
    """The bare minimum required to run a TTS model in inference mode."""

    input_encoder: train.spectrogram_model._data.InputEncoder
    spectrogram_model: SpectrogramModel
    signal_model: SignalModel


def main(
    spectrogram_checkpoint: pathlib.Path,
    signal_checkpoint: pathlib.Path,
    overwrite: bool = False,
):
    """Make a bundle of inference components from the SPECTROGRAM_CHECKPOINT and the
    SIGNAL_CHECKPOINT."""
    spec_ckpt = cast(train.spectrogram_model._worker.Checkpoint, load(spectrogram_checkpoint))
    sig_ckpt = cast(train.signal_model._worker.Checkpoint, load(signal_checkpoint))
    bundle = TTSBundle(*spec_ckpt.export(), sig_ckpt.export())
    save(TTS_BUNDLE_PATH, bundle, overwrite=overwrite)
    typer.echo(TTS_BUNDLE_PATH)


if __name__ == "__main__":  # pragma: no cover
    typer.run(main)

# NOTE: Needs to be imported before torch
import comet_ml  # noqa

import argparse

from src.hparams import set_hparams
from src.utils import Checkpoint
from src.bin.train.spectrogram_model.trainer import SpectrogramModelCheckpoint

if __name__ == '__main__':  # pragma: no cover
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--checkpoint', type=str)
    args = parser.parse_args()

    set_hparams()
    old_checkpoint = SpectrogramModelCheckpoint.from_path(args.checkpoint)
    Checkpoint(
        directory=old_checkpoint.directory,
        model=old_checkpoint.model,
        optimizer=old_checkpoint.optimizer,
        text_encoder=old_checkpoint.text_encoder,
        speaker_encoder=old_checkpoint.speaker_encoder,
        epoch=old_checkpoint.epoch,
        step=old_checkpoint.step,
        comet_ml_experiment_key=old_checkpoint.comet_ml_experiment_key).save()

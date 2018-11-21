""" Train spectrogram model.

 Example:
    $ python3 -m src.bin.train.spectrogram_model -n baseline;
"""
from pathlib import Path

import argparse
import logging
import time
import warnings

# LEARN MORE:
# https://stackoverflow.com/questions/40845304/runtimewarning-numpy-dtype-size-changed-may-indicate-binary-incompatibility
warnings.filterwarnings('ignore', message='numpy.dtype size changed')
warnings.filterwarnings('ignore', message='numpy.ufunc size changed')

# NOTE: Needs to be imported before torch
import comet_ml  # noqa

from torchnlp.text_encoders import CharacterEncoder
from torchnlp.text_encoders import IdentityEncoder

from src import datasets
from src.bin.train.spectrogram_model.trainer import Trainer
from src.hparams import add_config
from src.hparams import configurable
from src.hparams import set_hparams
from src.training_context_manager import TrainingContextManager
from src.utils import Checkpoint
from src.utils import parse_hparam_args
from src.utils import set_basic_logging_config

logger = logging.getLogger(__name__)


def _set_hparams():
    """ Set hyperparameters specific to the spectrogram model. """
    set_hparams()
    add_config({
        # SOURCE (Tacotron 2):
        # We use the Adam optimizer [29] with β1 = 0.9, β2 = 0.999, eps = 10−6
        # learning rate of 10−3
        # We also apply L2 regularization with weight 10−6
        'torch.optim.adam.Adam.__init__': {
            'eps': 10**-6,
            'weight_decay': 10**-5,
        }
    })


@configurable
def _get_dataset(dataset=datasets.lj_speech_dataset):
    return dataset


def main(run_one_liner,
         run_root=Path('experiments/spectrogram_model/'),
         comet_ml_project_name='spectrogram_model',
         checkpoint_path=None,
         reset_optimizer=False,
         hparams={},
         evaluate_every_n_epochs=1,
         save_checkpoint_every_n_epochs=10):
    """ Main module that trains a the spectrogram model saving checkpoints incrementally.

    Args:
        run_one_liner (str): One liner describing the experiment.
        run_root (str, optional): Directory to save experiments.
        comet_ml_project_name (str, optional): Project name to use with comet.ml.
        checkpoint_path (str, optional): Accepts a checkpoint path to load or empty string
            signaling to load the most recent checkpoint in ``run_root``.
        reset_optimizer (bool, optional): Given a checkpoint, resets the optimizer.
        hparams (dict, optional): Hparams to override default hparams.
        evaluate_every_n_epochs (int, optional)
        save_checkpoint_every_n_epochs (int, optional)
    """
    set_basic_logging_config()
    _set_hparams()
    add_config(hparams)

    checkpoint = (
        Checkpoint.most_recent(run_root / '**/*.pt')
        if checkpoint_path == '' else Checkpoint.from_path(checkpoint_path))

    if checkpoint is not None:
        step = checkpoint.step
        directory = checkpoint.directory.parent.parent
    else:
        step = 0
        directory = run_root / str(time.strftime('%b_%d/%H:%M:%S', time.localtime()))

    with TrainingContextManager(root_directory=directory, step=step) as context:
        logger.info('Using directory %s', directory)
        train, dev = _get_dataset()()
        text_encoder = CharacterEncoder(
            train['text']) if checkpoint is None else checkpoint.text_encoder
        speaker_encoder = IdentityEncoder(
            train['speaker']) if checkpoint is None else checkpoint.speaker_encoder

        # Load checkpointed values
        trainer_kwargs = {
            'text_encoder': text_encoder,
            'speaker_encoder': speaker_encoder,
            'comet_ml_project_name': comet_ml_project_name
        }
        if checkpoint is not None:
            logger.info('Loaded checkpoint %s', checkpoint.path)
            if reset_optimizer:
                logger.info('Ignoring checkpoint optimizer.')
                checkpoint.optimizer = None
            trainer_kwargs.update({
                'model': checkpoint.model,
                'optimizer': checkpoint.optimizer,
                'epoch': checkpoint.epoch,
                'step': checkpoint.step,
                'comet_ml_experiment_key': checkpoint.comet_ml_experiment_key,
            })

        trainer = Trainer(context.device, train, dev, **trainer_kwargs)
        trainer.comet_ml.log_other('one_liner', run_one_liner)

        # Training Loop
        while True:
            is_trial_run = trainer.step == step
            trainer.run_epoch(train=True, trial_run=is_trial_run)

            if trainer.epoch % evaluate_every_n_epochs == 0 or is_trial_run:
                trainer.run_epoch(train=False, trial_run=is_trial_run)

            if trainer.epoch % save_checkpoint_every_n_epochs == 0 or is_trial_run:
                Checkpoint(
                    directory=context.checkpoints_directory,
                    model=trainer.model,
                    optimizer=trainer.optimizer,
                    text_encoder=text_encoder,
                    speaker_encoder=speaker_encoder,
                    epoch=trainer.epoch,
                    step=trainer.step,
                    comet_ml_experiment_key=trainer.comet_ml.get_key()).save()

            print('–' * 100)


if __name__ == '__main__':  # pragma: no cover
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c',
        '--checkpoint',
        const='',
        type=str,
        default=None,
        action='store',
        nargs='?',
        help='Without a value, loads the most recent checkpoint;'
        'otherwise, expects a checkpoint file path.')
    parser.add_argument(
        '-l', '--one_liner', type=str, default=None, help='One liner describing the experiment')
    parser.add_argument(
        '-r', '--reset_optimizer', action='store_true', default=False, help='Reset optimizer.')
    args, unknown_args = parser.parse_known_args()
    hparams = parse_hparam_args(unknown_args)
    main(
        run_one_liner=args.one_liner,
        checkpoint_path=args.checkpoint,
        reset_optimizer=args.reset_optimizer,
        hparams=hparams)

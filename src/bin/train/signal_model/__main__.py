""" Train the signal model.

Example:
    $ python3 -m src.bin.train.signal_model -l="Linda baseline";
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

from src import datasets
from src.bin.train.signal_model.trainer import Trainer
from src.hparams import add_config
from src.hparams import configurable
from src.hparams import set_hparams
from src.training_context_manager import TrainingContextManager
from src.utils import Checkpoint
from src.utils import parse_hparam_args

logger = logging.getLogger(__name__)


def _set_hparams(more_hparams=None):
    """ Set hyperparameters for signal model training.

    Args:
        more_harpams (dict, optional): Additional hyperparameters to set.
    """
    set_hparams()

    add_config({
        # SOURCE (Tacotron 2):
        # We train with a batch size of 128 distributed across 32 GPUs with synchronous updates,
        # using the Adam optimizer with β1 = 0.9, β2 = 0.999, eps = 10−8 and a fixed learning rate
        # of 10−4
        # SOURCE (Deep Voice):
        # For training, we use the Adam optimization algorithm with β1 = 0.9, β2 = 0.999, ε = 10−8,
        # a batch size of 8, a learning rate of 10−3
        'torch.optim.adam.Adam.__init__': {
            'eps': 10**-8,
            'weight_decay': 0,
            'lr': 10**-3
        }
    })

    if more_hparams:
        add_config(more_hparams)


@configurable
def _get_dataset(dataset=datasets.lj_speech_dataset):
    return dataset()


def _train(trainer,
           context,
           evaluate_every_n_epochs=3,
           generate_every_n_epochs=30,
           save_checkpoint_every_n_epochs=15):
    """ Loop for training and periodically evaluating the model.

    Args:
        trainer (src.bin.train.signal_model.trainer.Trainer)
        context (src.training_context_manager.TrainingContextManager)
        evaluate_every_n_epochs (int, optional): Evaluate every ``evaluate_every_n_epochs`` epochs.
        generate_every_n_epochs (int, optional): Generate an audio sample every
            ``generate_every_n_epochs`` epochs.
        save_checkpoint_every_n_epochs (int, optional): Save a checkpoint every
            ``save_checkpoint_every_n_epochs`` epochs.
    """
    is_trial_run = True  # The first iteration is run as a ``trial_run``

    while True:
        trainer.run_epoch(train=True, trial_run=is_trial_run)

        if trainer.epoch % evaluate_every_n_epochs == 0 or is_trial_run:
            trainer.run_epoch(train=False, trial_run=is_trial_run)

        if trainer.epoch % generate_every_n_epochs == 0 or is_trial_run:
            trainer.visualize_inferred()

        if trainer.epoch % save_checkpoint_every_n_epochs == 0 or is_trial_run:
            trainer.save_checkpoint(context.checkpoints_directory)

        is_trial_run = False
        logger.info('-' * 100)


def main(run_name,
         comet_ml_project_name=None,
         run_tags=[],
         run_root=Path('experiments/signal_model/'),
         checkpoint=None,
         spectrogram_model_checkpoint_path=None,
         reset_optimizer=False,
         more_hparams={}):
    """ Main module that trains a the signal model saving checkpoints incrementally.

    Args:
        run_name (str): Name describing the experiment.
        comet_ml_project_name (str, optional): Project name to use with comet.ml.
        run_tags (list of str, optional): Tags describing the experiment.
        run_root (str, optional): Directory to save experiments.
        checkpoint_path (str or bool, optional): Accepts a checkpoint path to load or empty string
            signaling to load the most recent checkpoint in ``run_root``.
        spectrogram_model_checkpoint_path (str, optional): Checkpoint used to generate spectrogram
            from text as input to the signal model.
        reset_optimizer (bool, optional): Given a checkpoint, resets the optimizer.
        more_hparams (dict, optional): Hparams to override default hparams.
    """
    with TrainingContextManager() as context:
        logger.info('Name: %s', run_name)
        logger.info('Tags: %s', run_tags)

        _set_hparams(more_hparams)

        # Set the root directory and load checkpoint
        if checkpoint is not None and checkpoint:
            if isinstance(checkpoint, str):
                checkpoint = Checkpoint.from_path(checkpoint)
            elif isinstance(checkpoint, bool) and checkpoint:
                checkpoint = Checkpoint.most_recent(run_root / '**/*.pt')
            else:
                raise ValueError('Unable to load checkpoint.')

            context.set_context_root(checkpoint.directory.parent.parent, at_checkpoint=True)
            checkpoint.optimizer = None if reset_optimizer else checkpoint.optimizer
        else:
            root = run_root / str(time.strftime('%b_%d/%H:%M:%S', time.localtime())).lower()
            context.set_context_root(root)

        train, dev = _get_dataset()

        # Create trainer
        kwargs = {'device': context.device, 'train_dataset': train, 'dev_dataset': dev}
        if comet_ml_project_name is not None:
            kwargs['comet_ml_project_name'] = comet_ml_project_name
        if spectrogram_model_checkpoint_path is not None:
            kwargs['spectrogram_model_checkpoint_path'] = spectrogram_model_checkpoint_path
        trainer = (Trainer.from_checkpoint if checkpoint else Trainer)(**kwargs)

        trainer.comet_ml.set_name(run_name)
        trainer.comet_ml.add_tags(run_tags)
        trainer.comet_ml.log_other('directory', context.root_directory)

        _train(trainer, context)


if __name__ == '__main__':  # pragma: no cover
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c',
        '--checkpoint',
        const=True,
        type=str,
        default=None,
        action='store',
        nargs='?',
        help='Without a value ``-c``, loads the most recent checkpoint; '
        'otherwise, expects a checkpoint file path.')
    parser.add_argument(
        '-s',
        '--spectrogram_model_checkpoint',
        type=str,
        default=None,
        help=('Spectrogram model checkpoint path used to predicted spectrogram from '
              'text as input to the signal model.'))
    parser.add_argument(
        '-t', '--tags', default=[], action='append', help='List of tags for the experiment.')
    parser.add_argument(
        '-n', '--name', type=str, default=None, help='Name describing the experiment')
    parser.add_argument(
        '-r', '--reset_optimizer', action='store_true', default=False, help='Reset optimizer.')
    parser.add_argument(
        '-p',
        '--project_name',
        type=str,
        required=True,
        help='Comet.ML project for the experiment to use.')
    args, unknown_args = parser.parse_known_args()
    hparams = parse_hparam_args(unknown_args)
    main(
        run_name=args.name,
        run_tags=args.tags,
        comet_ml_project_name=args.project_name,
        checkpoint_path=args.checkpoint,
        spectrogram_model_checkpoint_path=args.spectrogram_model_checkpoint,
        reset_optimizer=args.reset_optimizer,
        more_hparams=hparams)

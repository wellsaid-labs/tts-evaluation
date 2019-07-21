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

# NOTE: Needs to be imported before torch
import comet_ml  # noqa

import torch

# NOTE: Some modules log on import; therefore, we first setup logging.
from src.environment import set_basic_logging_config

set_basic_logging_config()

from src.bin.train.signal_model.trainer import Trainer
from src.environment import assert_enough_disk_space
from src.environment import check_module_versions
from src.environment import ROOT_PATH
from src.environment import set_random_generator_state
from src.environment import set_seed
from src.hparams import add_config
from src.hparams import configurable
from src.hparams import ConfiguredArg
from src.hparams import parse_hparam_args
from src.hparams import set_hparams
from src.utils import Checkpoint
from src.utils import RecordStandardStreams
from src.visualize import CometML

logger = logging.getLogger(__name__)


def _set_hparams(more_hparams, checkpoint, comet_ml_project_name=None):
    """ Set hyperparameters for signal model training.

    Args:
        more_harpams (dict): Additional hyperparameters to set.
        checkpoint (src.utils.Checkpoint or None): Checkpoint to load random generator state from.
        comet_ml_project_name (str or None, optional)
    """
    set_hparams()

    comet_ml_project_name = (
        comet_ml_project_name if checkpoint is None else checkpoint.comet_ml_project_name)
    comet_ml_experiment_key = None if checkpoint is None else checkpoint.comet_ml_experiment_key

    add_config({
        'src.visualize.CometML': {
            'project_name': comet_ml_project_name,
            'experiment_key': comet_ml_experiment_key,
        },
    })
    add_config(more_hparams)

    set_seed()

    if checkpoint is not None and hasattr(checkpoint, 'random_generator_state'):
        set_random_generator_state(checkpoint.random_generator_state)


@configurable
def _get_dataset(dataset=ConfiguredArg()):
    return dataset()


def _train(trainer,
           evaluate_every_n_epochs=9,
           generate_every_n_evaluations=10,
           save_checkpoint_every_n_evaluations=5):
    """ Loop for training and periodically evaluating the model.

    Args:
        trainer (src.bin.train.signal_model.trainer.Trainer)
        evaluate_every_n_epochs (int, optional): Evaluate every ``evaluate_every_n_epochs`` epochs.
        generate_every_n_evaluations (int, optional): Generate an audio sample every
            ``generate_every_n_evaluations`` epochs.
        save_checkpoint_every_n_evaluations (int, optional): Save a checkpoint every
            ``save_checkpoint_every_n_evaluations`` epochs.
    """
    recent_checkpoint = None
    index = 0
    while True:
        is_trial_run = index == 0  # The first iteration is run as a ``trial_run``.
        trainer.run_epoch(train=True, trial_run=is_trial_run, num_epochs=evaluate_every_n_epochs)

        if index % save_checkpoint_every_n_evaluations == 0:
            trainer.save_checkpoint()
        else:
            # TODO: Consider using the GCP shutdown scripts via
            # https://haggainuchi.com/shutdown.html
            # NOTE: GCP shutdowns do not trigger `atexit`; therefore, it's useful to always save
            # a temporary checkpoint just in case.
            older_checkpoint = recent_checkpoint
            recent_checkpoint = trainer.save_checkpoint()
            if older_checkpoint is not None:
                older_checkpoint.unlink()  # Unlink only after `save_checkpoint` succeeds.

        trainer.run_epoch(train=False, trial_run=is_trial_run)

        if index % generate_every_n_evaluations == 0 or is_trial_run:
            trainer.visualize_inferred()

        index += 1
        logger.info('-' * 100)


def _time_label():
    return str(time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime())).lower()


def main(run_name,
         comet_ml_project_name=None,
         run_tags=[],
         run_root=ROOT_PATH / 'experiments' / 'signal_model' / _time_label(),
         checkpoints_directory=Path('checkpoints') / _time_label(),
         checkpoint=None,
         spectrogram_model_checkpoint_path=None,
         more_hparams={},
         device=torch.device('cuda')):
    """ Main module that trains a the signal model saving checkpoints incrementally.

    Args:
        run_name (str): Name describing the experiment.
        comet_ml_project_name (str, optional): Project name to use with comet.ml.
        run_tags (list of str, optional): Comet.ml experiment tags.
        run_root (str, optional): Directory to save experiments, unless a checkpoint is loaded.
        checkpoints_directory (str, optional): Directory to save checkpoints inside `run_root`.
        checkpoint (src.utils.Checkpoint, optional): Checkpoint or None.
        spectrogram_model_checkpoint_path (str, optional): Checkpoint used to generate spectrogram
            from text as input to the signal model.
        more_hparams (dict, optional): Hparams to override default hparams.
        device (torch.device): Primary device used for training.
    """
    recorder = RecordStandardStreams().start()
    _set_hparams(more_hparams, checkpoint, comet_ml_project_name)

    # Load `checkpoint`, setup `run_root`, and setup `checkpoints_directory`.
    run_root = run_root if checkpoint is None else checkpoint.directory.parent.parent
    if checkpoint is None:
        run_root.mkdir(parents=True)
    checkpoints_directory = run_root / checkpoints_directory
    checkpoints_directory.mkdir(parents=True)
    recorder.update(run_root)

    comet = CometML()
    logger.info('Name: %s', run_name)
    logger.info('Tags: %s', run_tags)
    comet.set_name(run_name)
    comet.add_tags(run_tags)
    comet.log_other('directory', run_root)

    train, dev = _get_dataset()

    # Create trainer
    kwargs = {
        'device': device,
        'train_dataset': train,
        'dev_dataset': dev,
        'checkpoints_directory': checkpoints_directory
    }
    if spectrogram_model_checkpoint_path is not None:
        kwargs['spectrogram_model_checkpoint_path'] = spectrogram_model_checkpoint_path
    if checkpoint is not None:
        kwargs['checkpoint'] = checkpoint
    trainer = (Trainer.from_checkpoint if checkpoint else Trainer)(**kwargs)

    _train(trainer)


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
        '-t',
        '--tags',
        default=[
            'batch_size=256',
            'lamb optimizer',
            'lr=2 * 10**-3',
            'rollback v4',
            'triangle LR schedule v3',
        ],
        action='append',
        help='List of tags for the experiment.')
    parser.add_argument(
        '-n', '--name', type=str, default=None, help='Name describing the experiment')
    parser.add_argument(
        '-r', '--reset_optimizer', action='store_true', default=False, help='Reset optimizer.')
    parser.add_argument(
        '-p', '--project_name', type=str, help='Comet.ML project for the experiment to use.')

    args, unparsed_args = parser.parse_known_args()

    # Pre-run checks on the `requirements.txt` and on the available disk space.
    check_module_versions()
    assert_enough_disk_space()

    if isinstance(args.checkpoint, str):
        args.checkpoint = Checkpoint.from_path(args.checkpoint)
    elif isinstance(args.checkpoint, bool) and args.checkpoint:
        args.checkpoint = Checkpoint.most_recent(ROOT_PATH / '**/*.pt')
    else:
        args.checkpoint = None

    if args.checkpoint is not None:
        # TODO: Remove coupling between `Trainer` and `checkpoint`.
        args.checkpoint.optimizer = None if args.reset_optimizer else args.checkpoint.optimizer
        args.project_name = args.checkpoint.comet_ml_project_name

    main(
        run_name=args.name,
        run_tags=args.tags,
        comet_ml_project_name=args.project_name,
        checkpoint=args.checkpoint,
        spectrogram_model_checkpoint_path=args.spectrogram_model_checkpoint,
        more_hparams=parse_hparam_args(unparsed_args))

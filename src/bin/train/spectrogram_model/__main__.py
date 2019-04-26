""" Train spectrogram model.

Example:
    $ python3 -m src.bin.train.spectrogram_model -l="Linda baseline";

Distributed Example:
    $ python3 -m third_party.launch src.bin.train.spectrogram_model -l="Linda distributed baseline";
    $ pkill -9 python3; nvidia-smi;

NOTE: The distributed example does clean up Python processes well; therefore, we kill all
``python3`` processes and check that ``nvidia-smi`` cache was cleared.
"""
from pathlib import Path

import argparse
import logging
import sys
import time
import warnings

# LEARN MORE:
# https://stackoverflow.com/questions/40845304/runtimewarning-numpy-dtype-size-changed-may-indicate-binary-incompatibility
warnings.filterwarnings('ignore', message='numpy.dtype size changed')
warnings.filterwarnings('ignore', message='numpy.ufunc size changed')

# NOTE: Needs to be imported before torch
import comet_ml  # noqa

from torch import multiprocessing

import torch

from src import datasets
from src.bin.train.spectrogram_model.trainer import Trainer
from src.hparams import add_config
from src.hparams import configurable
from src.hparams import set_hparams
from src.training_context_manager import TrainingContextManager
from src.utils import Checkpoint
from src.utils import parse_hparam_args

import src.distributed

logger = logging.getLogger(__name__)


def _set_hparams(more_hparams=None):
    """ Set hyperparameters for spectrogram model training.

    Args:
        more_harpams (dict, optional): Additional hyperparameters to set.
    """
    set_hparams()

    add_config({
        # SOURCE (Tacotron 2):
        # We use the Adam optimizer [29] with β1 = 0.9, β2 = 0.999, eps = 10−6
        # learning rate of 10−3
        # We also apply L2 regularization with weight 10−6
        'torch.optim.adam.Adam.__init__': {
            'eps': 10**-6,
            'weight_decay': 10**-6,
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
           evaluate_aligned_every_n_epochs=1,
           evaluate_inferred_every_n_epochs=5,
           save_checkpoint_every_n_epochs=5):
    """ Loop for training and periodically evaluating the model.

    Args:
        trainer (src.bin.train.spectrogram_model.trainer.Trainer)
        context (src.training_context_manager.TrainingContextManager)
        evaluate_aligned_every_n_epochs (int, optional)
        evaluate_inferred_every_n_epochs (int, optional)
        save_checkpoint_every_n_epochs (int, optional)
    """
    is_trial_run = True  # The first iteration is run as a ``trial_run``

    while True:
        trainer.run_epoch(train=True, trial_run=is_trial_run)

        if trainer.epoch % evaluate_aligned_every_n_epochs == 0 or is_trial_run:
            trainer.run_epoch(train=False, trial_run=is_trial_run)

        if trainer.epoch % evaluate_inferred_every_n_epochs == 0 or is_trial_run:
            trainer.run_epoch(train=False, infer=True, trial_run=is_trial_run)
            trainer.visualize_inferred()

        if ((trainer.epoch % save_checkpoint_every_n_epochs == 0 or is_trial_run) and
                src.distributed.is_master()):
            trainer.save_checkpoint(context.checkpoints_directory)

        is_trial_run = False
        logger.info('-' * 100)


def main(run_name,
         comet_ml_project_name=None,
         run_tags=[],
         run_root=Path('experiments/spectrogram_model/'),
         checkpoint=None,
         reset_optimizer=False,
         more_hparams={},
         device_index=None):
    """ Main module that trains a the spectrogram model saving checkpoints incrementally.

    Args:
        run_name (str): Name describing the experiment.
        comet_ml_project_name (str, optional): Project name to use with comet.ml.
        run_tags (list of str, optional): Comet.ml experiment tags.
        run_root (str, optional): Directory to save experiments.
        checkpoint (str or bool, optional): Accepts a checkpoint path to load or bool
            signaling to load the most recent checkpoint in ``run_root``.
        reset_optimizer (bool, optional): Given a checkpoint, resets the optimizer.
        more_hparams (dict, optional): Hparams to override default hparams.
        device_index (int, optional): Index of the GPU device to use.
    """
    device = torch.device('cuda') if device_index is None else torch.device('cuda', device_index)
    with TrainingContextManager(device=device) as context:
        logger.info('Name: %s', run_name)
        logger.info('Tags: %s', run_tags)

        context.init_distributed()

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
            if src.distributed.is_initialized():
                root = Path(src.distributed.broadcast_string(str(root)))
            context.set_context_root(root)

        train, dev = _get_dataset()

        # Create trainer
        kwargs = {'device': device, 'train_dataset': train, 'dev_dataset': dev}
        if comet_ml_project_name is not None:
            kwargs['comet_ml_project_name'] = comet_ml_project_name
        if checkpoint:
            kwargs['checkpoint'] = checkpoint
        trainer = (Trainer.from_checkpoint if checkpoint else Trainer)(**kwargs)

        # TODO: Consider ignoring ``add_tags`` if Checkpoint is loaded; or consider saving in the
        # checkpoint the ``name`` and ``tags``; or consider fetching tags from the Comet.ML API.
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
        '-n', '--name', type=str, default=None, help='Name describing the experiment')
    parser.add_argument(
        '-p', '--project_name', type=str, help='Comet.ML project for the experiment to use.')
    parser.add_argument(
        '-t',
        '--tags',
        default=['detached post_net', 'masked conv', 'post_net dropout=0', 'weight_decay=10**-6'],
        action='append',
        help='List of tags for the experiment.')
    parser.add_argument(
        '-r', '--reset_optimizer', action='store_true', default=False, help='Reset optimizer.')
    # LEARN MORE: https://pytorch.org/docs/stable/distributed.html
    parser.add_argument(
        '--local_rank',
        type=int,
        default=None,
        help='Argument provided by distributed launch utility.')
    args, unknown_args = parser.parse_known_args()
    hparams = parse_hparam_args(unknown_args)

    if args.local_rank is not None:
        # Python version must be 3.6.6 or higher
        assert sys.version_info >= (3, 6, 6)
        try:
            # LEARN MORE:
            # https://pytorch.org/docs/stable/nn.html?highlight=distributeddataparallel#torch.nn.parallel.DistributedDataParallel
            # https://github.com/tqdm/tqdm/issues/611#issuecomment-423113285
            multiprocessing.set_start_method('spawn')
        except RuntimeError:
            pass

    main(
        run_name=args.name,
        run_tags=args.tags,
        comet_ml_project_name=args.project_name,
        checkpoint=args.checkpoint,
        reset_optimizer=args.reset_optimizer,
        more_hparams=hparams,
        device_index=args.local_rank)

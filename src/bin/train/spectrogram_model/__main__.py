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
from torchnlp.text_encoders import CharacterEncoder
from torchnlp.text_encoders import IdentityEncoder

import torch

from src import datasets
from src.bin.train.spectrogram_model.trainer import Trainer
from src.bin.train.spectrogram_model.trainer import SpectrogramModelCheckpoint
from src.hparams import add_config
from src.hparams import configurable
from src.hparams import set_hparams
from src.training_context_manager import TrainingContextManager
from src.utils import parse_hparam_args
from src.utils import set_basic_logging_config

import src.distributed

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
         comet_ml_project_name='spectrogram-model-baselines',
         checkpoint_path=None,
         reset_optimizer=False,
         hparams={},
         evaluate_every_n_epochs=1,
         save_checkpoint_every_n_epochs=10,
         distributed_rank=None,
         distributed_backend='nccl'):
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
        distributed_rank (int, optional): Index of the GPU device to use in distributed system.
        distributed_backend (str, optional): Name of the backend to use.
    """
    device = None
    if distributed_rank is not None:
        torch.distributed.init_process_group(backend=distributed_backend)
        device = torch.device('cuda', distributed_rank)
        torch.cuda.set_device(device.index)  # Must run before using distributed tensors

    set_basic_logging_config(device)
    _set_hparams()
    add_config(hparams)

    checkpoint = (
        SpectrogramModelCheckpoint.most_recent(run_root / '**/*.pt')
        if checkpoint_path == '' else SpectrogramModelCheckpoint.from_path(checkpoint_path))

    if checkpoint is not None:
        step = checkpoint.step
        directory = checkpoint.directory.parent.parent
    else:
        step = 0
        directory = run_root / str(time.strftime('%b_%d/%H:%M:%S', time.localtime()))
        if torch.distributed.is_initialized():
            directory = Path(src.distributed.broadcast_string(str(directory)))

    with TrainingContextManager(root_directory=directory, step=step, device=device) as context:
        logger.info('Using directory %s', directory)
        train, dev = _get_dataset()()

        # Load checkpointed values
        trainer_kwargs = {
            'comet_ml_project_name': comet_ml_project_name,
        }
        if checkpoint is not None:
            if reset_optimizer:
                logger.info('Ignoring checkpoint optimizer.')
            logger.info('Loaded checkpoint %s', checkpoint.path)
            trainer_kwargs.update({
                'model': checkpoint.model,
                'optimizer': None if reset_optimizer else checkpoint.optimizer,
                'epoch': checkpoint.epoch,
                'step': checkpoint.step,
                'comet_ml_experiment_key': checkpoint.comet_ml_experiment_key,
                'text_encoder': checkpoint.text_encoder,
                'speaker_encoder': checkpoint.speaker_encoder
            })
        else:
            trainer_kwargs.update({
                'text_encoder': CharacterEncoder(train['text']),
                'speaker_encoder': IdentityEncoder(train['speaker']),
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
                if not torch.distributed.is_initialized() or src.distributed.is_master():
                    SpectrogramModelCheckpoint(
                        directory=context.checkpoints_directory,
                        model_state_dict=trainer.model.module.state_dict()
                        if isinstance(trainer.model, torch.nn.parallel.DistributedDataParallel) else
                        trainer.model.state_dict(),
                        optimizer_state_dict=trainer.optimizer.state_dict(),
                        text_encoder=trainer_kwargs['text_encoder'],
                        speaker_encoder=trainer_kwargs['speaker_encoder'],
                        epoch=trainer.epoch,
                        step=trainer.step,
                        comet_ml_experiment_key=trainer.comet_ml.get_key()).save()

            logger.info('-' * 100)


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
        run_one_liner=args.one_liner,
        checkpoint_path=args.checkpoint,
        reset_optimizer=args.reset_optimizer,
        hparams=hparams,
        distributed_rank=args.local_rank)

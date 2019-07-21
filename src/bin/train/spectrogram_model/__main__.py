""" Train spectrogram model.

Example:

    $ PYTHONPATH='.' python3 src/bin/train/spectrogram_model/__main__.py  \
        -p "stft-baselines" \
        -n "Multispeaker v3 baseline";
    $ pkill -9 python3; nvidia-smi;

NOTE: The distributed example does clean up Python processes well; therefore, we kill all
``python3`` processes and check that ``nvidia-smi`` cache was cleared.
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

from src.bin.train.spectrogram_model.trainer import Trainer
from src.datasets import add_spectrogram_column
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
from src.utils import cache_on_disk_tensor_shapes
from src.utils import Checkpoint
from src.utils import RecordStandardStreams
from src.visualize import CometML

import src.distributed

logger = logging.getLogger(__name__)


def _set_hparams(more_hparams, checkpoint, comet_ml_project_name=None,
                 comet_ml_experiment_key=None):
    """ Set hyperparameters for spectrogram model training.

    Args:
        more_harpams (dict): Additional hyperparameters to set.
        checkpoint (src.utils.Checkpoint): Checkpoint to load random generator state from.
        comet_ml_project_name (str or None, optional)
        comet_ml_experiment_key (str or None, optional)
    """
    set_hparams()

    comet_ml_project_name = (
        comet_ml_project_name if checkpoint is None else checkpoint.comet_ml_project_name)
    comet_ml_experiment_key = (
        comet_ml_experiment_key if checkpoint is None else checkpoint.comet_ml_experiment_key)

    add_config({
        # SOURCE (Tacotron 2):
        # We use the Adam optimizer [29] with β1 = 0.9, β2 = 0.999, eps = 10−6
        # learning rate of 10−3
        # We also apply L2 regularization with weight 10−6
        'torch.optim.adam.Adam.__init__': {
            'eps': 10**-6,
            'weight_decay': 10**-6,
            'lr': 10**-3
        },
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


def _train(device_index,
           run_root,
           checkpoints_directory,
           checkpoint,
           train_dataset,
           dev_dataset,
           comet_ml_project_name,
           comet_ml_experiment_key,
           more_hparams,
           evaluate_aligned_every_n_epochs=1,
           evaluate_inferred_every_n_epochs=5,
           save_checkpoint_every_n_epochs=5,
           distributed_backend='nccl',
           distributed_init_method='tcp://127.0.0.1:29500'):
    """ Loop for training and periodically evaluating the model.

    Args:
        device_index (int)
        run_root (Path): Directory to save experiment.
        checkpoints_directory (Path): Directory to save checkpoints.
        checkpoint (src.utils.Checkpoint): Loaded `Checkpoint` or None.
        train_dataset (iterable)
        dev_dataset (iterable)
        comet_ml_project_name (str): Project name to use with comet.ml.
        comet_ml_experiment_key (str): Experiment key to use with comet.ml.
        more_hparams (dict): Hparams to override default hparams.
    """
    recorder = RecordStandardStreams().start()
    # Initiate distributed environment, learn more:
    # https://pytorch.org/tutorials/intermediate/dist_tuto.htm
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    torch.distributed.init_process_group(
        backend=distributed_backend,
        rank=device_index,
        world_size=torch.cuda.device_count(),
        init_method=distributed_init_method)
    device = torch.device('cuda', device_index)
    torch.cuda.set_device(device)

    _set_hparams(more_hparams, checkpoint, comet_ml_project_name, comet_ml_experiment_key)
    recorder.update(run_root)

    trainer_kwargs = {
        'device': device,
        'train_dataset': train_dataset,
        'dev_dataset': dev_dataset,
        'checkpoints_directory': checkpoints_directory
    }
    if checkpoint is not None:
        trainer_kwargs['checkpoint'] = checkpoint
    trainer = (Trainer.from_checkpoint if checkpoint else Trainer)(**trainer_kwargs)

    is_trial_run = True  # The first iteration is run as a ``trial_run``
    recent_checkpoint = None
    while True:
        trainer.run_epoch(train=True, trial_run=is_trial_run)

        if trainer.epoch % save_checkpoint_every_n_epochs == 0 and src.distributed.is_master():
            trainer.save_checkpoint()
        elif src.distributed.is_master():
            # NOTE: GCP shutdowns do not trigger `atexit`; therefore, it's useful to always save
            # a temporary checkpoint just in case.
            older_checkpoint = recent_checkpoint
            recent_checkpoint = trainer.save_checkpoint()
            if older_checkpoint is not None:
                older_checkpoint.unlink()

        if trainer.epoch % evaluate_aligned_every_n_epochs == 0 or is_trial_run:
            trainer.run_epoch(train=False, trial_run=is_trial_run)

        if trainer.epoch % evaluate_inferred_every_n_epochs == 0 or is_trial_run:
            trainer.run_epoch(train=False, infer=True, trial_run=is_trial_run)
            trainer.visualize_inferred()

        is_trial_run = False
        logger.info('-' * 100)


def _time_label():
    return str(time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime())).lower()


def main(run_name,
         comet_ml_project_name=None,
         run_tags=[],
         run_root=ROOT_PATH / 'experiments' / 'spectrogram_model' / _time_label(),
         checkpoints_directory=Path('checkpoints') / _time_label(),
         checkpoint=None,
         more_hparams={}):
    """ Main module that trains a the spectrogram model saving checkpoints incrementally.

    Args:
        run_name (str): Name describing the experiment.
        comet_ml_project_name (str, optional): Project name to use with comet.ml.
        run_tags (list of str, optional): Comet.ml experiment tags.
        run_root (str, optional): Directory to save experiments, unless a checkpoint is loaded.
        checkpoints_directory (str, optional): Directory to save checkpoints inside `run_root`.
        checkpoint (src.utils.Checkpoint, optional): Checkpoint or None.
        more_hparams (dict, optional): Hparams to override default hparams.
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

    # TODO: Consider ignoring ``add_tags`` if Checkpoint is loaded; or consider saving in the
    # checkpoint the ``name`` and ``tags``; or consider fetching tags from the Comet.ML API.
    comet = CometML()
    logger.info('Name: %s', run_name)
    logger.info('Tags: %s', run_tags)
    comet.set_name(run_name)
    comet.add_tags(run_tags)
    comet.log_other('directory', run_root)

    train_dataset, dev_dataset = _get_dataset()
    # NOTE: Preprocessing is faster to compute outside of distributed environment.
    train_dataset = add_spectrogram_column(train_dataset)
    dev_dataset = add_spectrogram_column(dev_dataset)
    # TODO: Consider supporting `Tensor` as well as `OnDiskTensor`.
    cache_on_disk_tensor_shapes([e.spectrogram for e in train_dataset] +
                                [e.spectrogram for e in dev_dataset])

    num_cuda_devices = torch.cuda.device_count()
    torch.multiprocessing.spawn(
        _train,
        nprocs=num_cuda_devices,
        args=(
            run_root,
            checkpoints_directory,
            checkpoint,
            train_dataset,
            dev_dataset,
            comet_ml_project_name,
            comet.get_key(),
            more_hparams,
        ))


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
    parser.add_argument('-n', '--name', type=str, default=None, help='Name of a new experiment.')
    parser.add_argument(
        '-p',
        '--project_name',
        type=str,
        default=None,
        help='Name of the comet.ml project to store a new experiment in.')
    # NOTE: The baseline tags summarize changes in the current repository.
    parser.add_argument(
        '-t',
        '--tags',
        default=[
            'detached post_net', 'masked conv', 'post_net dropout=0', 'weight_decay=10**-6',
            'no elliot', 'no numbers'
        ],
        action='append',
        help='List of tags for a new experiments.')
    parser.add_argument(
        '-r',
        '--reset_optimizer',
        action='store_true',
        default=False,
        help='Resets the checkpoint optimizer if provided.')

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
        args.checkpoint.optimizer = None if args.reset_optimizer else args.checkpoint.optimizer
        args.project_name = args.checkpoint.comet_ml_project_name

    main(
        run_name=args.name,
        run_tags=args.tags,
        comet_ml_project_name=args.project_name,
        checkpoint=args.checkpoint,
        more_hparams=parse_hparam_args(unparsed_args))

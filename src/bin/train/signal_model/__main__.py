""" Train the signal model.

Example:

    $ pkill -9 python; nvidia-smi; PYTHONPATH=. python src/bin/train/signal_model/__main__.py

NOTE: The distributed example does clean up Python processes well; therefore, we kill all
``python`` processes and check that ``nvidia-smi`` cache was cleared.
"""
from pathlib import Path

import argparse
import logging
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
from src.datasets import add_predicted_spectrogram_column
from src.datasets import add_spectrogram_column
from src.environment import assert_enough_disk_space
from src.environment import check_module_versions
from src.environment import EXPERIMENTS_PATH
from src.environment import set_random_generator_state
from src.environment import set_seed
from src.hparams import add_config
from src.hparams import configurable
from src.hparams import ConfiguredArg
from src.hparams import parse_hparam_args
from src.hparams import set_hparams
from src.utils import bash_time_label
from src.utils import cache_on_disk_tensor_shapes
from src.utils import Checkpoint
from src.utils import RecordStandardStreams
from src.visualize import CometML

import src.distributed

logger = logging.getLogger(__name__)


def _set_hparams(more_hparams, checkpoint, comet_ml_project_name=None,
                 comet_ml_experiment_key=None):
    """ Set hyperparameters for signal model training.

    Args:
        more_harpams (dict): Additional hyperparameters to set.
        checkpoint (src.utils.Checkpoint or None): Checkpoint to load random generator state from.
        comet_ml_project_name (str or None, optional)
    """
    set_hparams()

    comet_ml_project_name = (
        comet_ml_project_name if checkpoint is None else checkpoint.comet_ml_project_name)
    comet_ml_experiment_key = (
        comet_ml_experiment_key if checkpoint is None else checkpoint.comet_ml_experiment_key)

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


def _train(device_index,
           run_root,
           checkpoints_directory,
           checkpoint,
           spectrogram_model_checkpoint,
           train_dataset,
           dev_dataset,
           comet_ml_project_name,
           comet_ml_experiment_key,
           more_hparams,
           evaluate_every_n_epochs=9,
           generate_every_n_evaluations=1,
           save_checkpoint_every_n_evaluations=5,
           distributed_backend='nccl',
           distributed_init_method='tcp://127.0.0.1:29500'):
    """ Loop for training and periodically evaluating the model.

    Args:
        device_index (int)
        run_root (Path): Directory to save experiment.
        checkpoints_directory (Path): Directory to save checkpoints.
        checkpoint (src.utils.Checkpoint): Loaded `Checkpoint` or None.
        spectrogram_model_checkpoint (src.utils.Checkpoint)
        train_dataset (iterable)
        dev_dataset (iterable)
        comet_ml_project_name (str): Project name to use with comet.ml.
        comet_ml_experiment_key (str): Experiment key to use with comet.ml.
        more_hparams (dict): Hparams to override default hparams.
        evaluate_every_n_epochs (int, optional): Evaluate every ``evaluate_every_n_epochs`` epochs.
        generate_every_n_evaluations (int, optional): Generate an audio sample every
            ``generate_every_n_evaluations`` epochs.
        save_checkpoint_every_n_evaluations (int, optional): Save a checkpoint every
            ``save_checkpoint_every_n_evaluations`` epochs.
        distributed_backend (str)
        distributed_init_method (str)
    """
    # TODO: Consider naming the logs based on the time they are started for sorting.
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

    logger.info('Worker %d started.', torch.distributed.get_rank())

    _set_hparams(more_hparams, checkpoint, comet_ml_project_name, comet_ml_experiment_key)
    recorder.update(run_root)

    trainer_kwargs = {
        'device': device,
        'train_dataset': train_dataset,
        'dev_dataset': dev_dataset,
        'checkpoints_directory': checkpoints_directory
    }
    if spectrogram_model_checkpoint is not None:
        trainer_kwargs['spectrogram_model_checkpoint'] = spectrogram_model_checkpoint
    if checkpoint is not None:
        trainer_kwargs['checkpoint'] = checkpoint
    trainer = (Trainer.from_checkpoint if checkpoint else Trainer)(**trainer_kwargs)

    recent_checkpoint = None
    index = 0
    while True:
        is_trial_run = index == 0  # The first iteration is run as a ``trial_run``.

        for _ in range(1 if is_trial_run else evaluate_every_n_epochs):
            trainer.run_epoch(train=True, trial_run=is_trial_run)

        if index % save_checkpoint_every_n_evaluations == 0 and src.distributed.is_master():
            trainer.save_checkpoint()
        elif src.distributed.is_master():
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


def main(run_name=None,
         comet_ml_project_name=None,
         run_tags=[],
         run_root=EXPERIMENTS_PATH / 'signal_model' / bash_time_label(),
         checkpoints_directory=Path('checkpoints') / bash_time_label(),
         checkpoint=None,
         spectrogram_model_checkpoint=None,
         more_hparams={},
         device=torch.device('cuda')):
    """ Main module that trains a the signal model saving checkpoints incrementally.

    TODO: Test this module.

    Args:
        run_name (str, optional): Name of the experiment.
        comet_ml_project_name (str, optional): Project name to use with comet.ml.
        run_tags (list of str, optional): Comet.ml experiment tags.
        run_root (str, optional): Directory to save experiments, unless a checkpoint is loaded.
        checkpoints_directory (str, optional): Directory to save checkpoints inside `run_root`.
        checkpoint (src.utils.Checkpoint, optional): Checkpoint or None.
        spectrogram_model_checkpoint (str, optional): Checkpoint used to generate spectrogram
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
    add_config({'src.visualize.CometML.experiment_key': comet.get_key()})
    if run_name is not None:
        logger.info('Name: %s', run_name)
        comet.set_name(run_name)
    logger.info('Tags: %s', run_tags)
    comet.add_tags(run_tags)
    comet.log_other('directory', run_root)

    def _preprocess_data(data):
        data = add_spectrogram_column(data)
        cache_on_disk_tensor_shapes([r.spectrogram for r in data])
        if spectrogram_model_checkpoint is not None:
            data = add_predicted_spectrogram_column(data, spectrogram_model_checkpoint, device)
            cache_on_disk_tensor_shapes([r.predicted_spectrogram for r in data])
        return data

    train_dataset, dev_dataset = _get_dataset()
    train_dataset = _preprocess_data(train_dataset)
    dev_dataset = _preprocess_data(dev_dataset)

    num_cuda_devices = torch.cuda.device_count()
    # NOTE (michael): Without this assert, when `nprocs` is zero, `torch.multiprocessing.spawn`
    # crashes in a nondescript way.
    assert num_cuda_devices > 0, 'Unable to find CUDA devices.'
    torch.multiprocessing.spawn(
        _train,
        nprocs=num_cuda_devices,
        args=(
            run_root,
            checkpoints_directory,
            checkpoint,
            spectrogram_model_checkpoint,
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
            'rollback v5',
            'triangle LR schedule v3',
        ],
        action='append',
        help='List of tags for the experiment.')
    parser.add_argument('-n', '--name', type=str, default=None, help='Name of the experiment.')
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
        args.checkpoint = Checkpoint.most_recent(EXPERIMENTS_PATH / '**/*.pt')
    else:
        args.checkpoint = None

    if args.spectrogram_model_checkpoint is not None:
        args.spectrogram_model_checkpoint = Checkpoint.from_path(args.spectrogram_model_checkpoint)

    if args.checkpoint is not None:
        # TODO: Remove coupling between `Trainer` and `checkpoint`.
        args.checkpoint.optimizer = None if args.reset_optimizer else args.checkpoint.optimizer
        args.project_name = args.checkpoint.comet_ml_project_name

    main(
        run_name=args.name,
        run_tags=args.tags,
        comet_ml_project_name=args.project_name,
        checkpoint=args.checkpoint,
        spectrogram_model_checkpoint=args.spectrogram_model_checkpoint,
        more_hparams=parse_hparam_args(unparsed_args))

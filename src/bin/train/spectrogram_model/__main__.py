""" Train spectrogram model.

Example:

    $ pkill -9 python; nvidia-smi; PYTHONPATH=. python src/bin/train/spectrogram_model/__main__.py

NOTE: The distributed example does clean up Python processes well; therefore, we kill all
``python`` processes and check that ``nvidia-smi`` cache was cleared.
"""
import argparse
import logging
import warnings

# NOTE: Needs to be imported before torch
import comet_ml  # noqa

from hparams import add_config
from hparams import configurable
from hparams import HParam
from hparams import HParams
from hparams import parse_hparam_args
from torchnlp.random import set_random_generator_state

# LEARN MORE:
# https://stackoverflow.com/questions/40845304/runtimewarning-numpy-dtype-size-changed-may-indicate-binary-incompatibility
warnings.filterwarnings('ignore', message='numpy.dtype size changed')
warnings.filterwarnings('ignore', message='numpy.ufunc size changed')

import torch

from src.bin.train.spectrogram_model.trainer import Trainer
from src.datasets import add_spectrogram_column
from src.environment import assert_enough_disk_space
from src.environment import check_module_versions
from src.environment import set_basic_logging_config
from src.environment import set_seed
from src.environment import SPECTROGRAM_MODEL_EXPERIMENTS_PATH
from src.hparams import set_hparams
from src.utils import bash_time_label
from src.utils import cache_on_disk_tensor_shapes
from src.utils import Checkpoint
from src.utils import RecordStandardStreams
from src.visualize import CometML

import src.distributed


def _set_hparams(more_hparams, checkpoint):
    """ Set hyperparameters for spectrogram model training.

    Args:
        more_harpams (dict): Additional hyperparameters to set.
        checkpoint (src.utils.Checkpoint): Checkpoint to load random generator state from.
    """
    set_hparams()
    add_config({
        # SOURCE (Tacotron 2):
        # We use the Adam optimizer [29] with β1 = 0.9, β2 = 0.999, eps = 10−6
        # learning rate of 10−3
        # We also apply L2 regularization with weight 10−6
        'torch.optim.adam.Adam.__init__': HParams(eps=10**-6, weight_decay=10**-6, lr=10**-3)
    })
    add_config(more_hparams)
    set_seed()
    if checkpoint is not None and hasattr(checkpoint, 'random_generator_state'):
        set_random_generator_state(checkpoint.random_generator_state)


@configurable
def _get_dataset(dataset=HParam()):
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
           evaluate_inferred_every_n_epochs=3,
           save_checkpoint_every_n_epochs=3,
           distributed_backend='nccl',
           distributed_init_method='tcp://127.0.0.1:29500'):
    """ Loop for training and periodically evaluating the model.

    TODO: Discuss how `distributed_init_method` and `distributed_backend` were choosen.

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
        evaluate_aligned_every_n_epochs (int, optional)
        evaluate_inferred_every_n_epochs (int, optional)
        save_checkpoint_every_n_epochs (int, optional)
        distributed_backend (str, optional)
        distributed_init_method (str, optional)
    """
    set_basic_logging_config(device_index)
    logger = logging.getLogger(__name__)

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

    comet = CometML(
        project_name=comet_ml_project_name,
        experiment_key=comet_ml_experiment_key,
        disabled=not src.distributed.is_master(),
        auto_output_logging=False)

    logger.info('Worker %d started.', torch.distributed.get_rank())

    _set_hparams(more_hparams, checkpoint)

    trainer_kwargs = {
        'device': device,
        'train_dataset': train_dataset,
        'dev_dataset': dev_dataset,
        'checkpoints_directory': checkpoints_directory,
        'comet_ml': comet,
    }
    if checkpoint is not None:
        trainer_kwargs['checkpoint'] = checkpoint
    trainer = (Trainer.from_checkpoint if checkpoint else Trainer)(**trainer_kwargs)

    is_trial_run = True  # The first iteration is run as a ``trial_run``
    while True:
        trainer.run_epoch(train=True, trial_run=is_trial_run)

        if trainer.epoch % save_checkpoint_every_n_epochs == 0 and src.distributed.is_master():
            trainer.save_checkpoint()

        if trainer.epoch % evaluate_aligned_every_n_epochs == 0 or is_trial_run:
            trainer.run_epoch(train=False, trial_run=is_trial_run)

        if trainer.epoch % evaluate_inferred_every_n_epochs == 0 or is_trial_run:
            trainer.run_epoch(train=False, infer=True, trial_run=is_trial_run)
            trainer.visualize_inferred()

        is_trial_run = False
        logger.info('-' * 100)


def main(experiment_name=None,
         comet_ml_experiment_key=None,
         comet_ml_project_name=None,
         experiment_tags=[],
         experiment_root=SPECTROGRAM_MODEL_EXPERIMENTS_PATH / bash_time_label(),
         run_name='RUN_' + bash_time_label(add_pid=False),
         checkpoints_directory_name='checkpoints',
         checkpoint=None,
         more_hparams={}):
    """ Main module that trains a the spectrogram model saving checkpoints incrementally.

    Args:
        experiment_name (str, optional): Name of the experiment.
        comet_ml_experiment_key (str, optional): Experiment key to use with comet.ml.
        comet_ml_project_name (str, optional): Project name to use with comet.ml.
        experiment_tags (list of str, optional): Comet.ml experiment tags.
        experiment_root (str, optional): Directory to save experiments, unless a checkpoint is
            loaded.
        run_name (str, optional): The name of the run.
        checkpoints_directory_name (str, optional): The name of the checkpoint directory.
        checkpoint (src.utils.Checkpoint, optional): Checkpoint or None.
        more_hparams (dict, optional): Hparams to override default hparams.
    """
    set_basic_logging_config()
    logger = logging.getLogger(__name__)
    comet = CometML(project_name=comet_ml_project_name, experiment_key=comet_ml_experiment_key)
    recorder = RecordStandardStreams().start()
    _set_hparams(more_hparams, checkpoint)

    # Load `checkpoint`, setup `run_root`, and setup `checkpoints_directory`.
    experiment_root = experiment_root if checkpoint is None else checkpoint.directory.parent.parent
    run_root = experiment_root / run_name
    run_root.mkdir(parents=checkpoint is None)
    checkpoints_directory = run_root / checkpoints_directory_name
    checkpoints_directory.mkdir()
    recorder.update(run_root, log_filename='run.log')

    # TODO: Consider ignoring ``add_tags`` if Checkpoint is loaded; or consider saving in the
    # checkpoint the ``name`` and ``tags``; or consider fetching tags from the Comet.ML API.
    if experiment_name is not None:
        logger.info('Name: %s', experiment_name)
        comet.set_name(experiment_name)
    logger.info('Tags: %s', experiment_tags)
    comet.add_tags(experiment_tags)
    comet.log_other('directory', str(run_root))

    train_dataset, dev_dataset = _get_dataset()
    # NOTE: Preprocessing is faster to compute outside of distributed environment.
    train_dataset = add_spectrogram_column(train_dataset)
    dev_dataset = add_spectrogram_column(dev_dataset)
    cache_on_disk_tensor_shapes([e.spectrogram for e in train_dataset] +
                                [e.spectrogram for e in dev_dataset])

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
            train_dataset,
            dev_dataset,
            comet_ml_project_name,
            comet.get_key(),
            more_hparams,
        ))


if __name__ == '__main__':  # pragma: no cover
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--checkpoint',
        const=True,
        type=str,
        default=None,
        action='store',
        nargs='?',
        help='Without a value, this loads the most recent checkpoint; '
        'otherwise, expects a checkpoint file path.')
    parser.add_argument('--name', type=str, default=None, help='Name of the experiment.')
    parser.add_argument(
        '--project_name',
        type=str,
        default=None,
        help='Name of the comet.ml project to store a new experiment in.')
    # NOTE: The baseline tags summarize changes in the current repository.
    parser.add_argument(
        '--tags',
        default=[
            '1024_frame_size', '16_bit_audio', 'amsgrad=False', 'dataset_filter', 'db_scale',
            'iso226_weighting', 'lower_hertz_20', 'min_decibel_50', 'min_padding',
            'power_before_mel_scale', 'pytorch_1_4', 'pytorch_stft', 'zero_go_frame',
            'pad_before_trim', 'larger_half_gaussian', 'lower_reached_max', '500_step_warmup',
            '0_01_spectrogram_loss', '10_output_scalar', 'ema', '1024_zero_padding',
            'predict_inital', 'pre_net_layer_norm', 'no_speaker_embed_dropout'
        ],
        nargs='+',
        help='List of tags for a new experiments.')
    parser.add_argument(
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
        args.checkpoint = Checkpoint.most_recent(SPECTROGRAM_MODEL_EXPERIMENTS_PATH / '**/*.pt')
    else:
        args.checkpoint = None

    comet_ml_experiment_key = None
    if args.checkpoint is not None:
        args.checkpoint.optimizer = None if args.reset_optimizer else args.checkpoint.optimizer
        args.project_name = args.checkpoint.comet_ml_project_name
        comet_ml_experiment_key = args.checkpoint.comet_ml_experiment_key

    main(
        experiment_name=args.name,
        experiment_tags=args.tags,
        comet_ml_experiment_key=comet_ml_experiment_key,
        comet_ml_project_name=args.project_name,
        checkpoint=args.checkpoint,
        more_hparams=parse_hparam_args(unparsed_args))

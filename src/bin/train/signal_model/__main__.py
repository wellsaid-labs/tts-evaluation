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
from src.hparams import log_config
from src.hparams import set_hparams
from src.training_context_manager import TrainingContextManager
from src.utils import Checkpoint
from src.utils import parse_hparam_args
from src.utils import set_basic_logging_config

logger = logging.getLogger(__name__)


def _set_hparams():
    """ Set auxillary hyperparameters specific to the signal model. """
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


@configurable
def _get_dataset(dataset=datasets.lj_speech_dataset):
    return dataset


def main(run_name,
         run_root='experiments/signal_model/',
         checkpoint_path=None,
         spectrogram_model_checkpoint_path=None,
         reset_optimizer=False,
         hparams={},
         evaluate_every_n_epochs=15):
    """ Main module that trains a the signal model saving checkpoints incrementally.

    Args:
        run_name (str): Variable used in experiment directory path ``{run_root}/MM_DD/{run_name}/``.
        run_root (str, optional): Directory to save experiments.
        checkpoint_path (str, optional): Accepts a checkpoint path to load or empty string
            signaling to load the most recent checkpoint in ``run_root``.
        spectrogram_model_checkpoint_path (str, optional): Checkpoint used to generate spectrogram
            from text as input to the signal model.
        reset_optimizer (bool, optional): Given a checkpoint, resets the optimizer.
        hparams (dict, optional): Hparams to override default hparams.
        evaluate_every_n_epochs (int, optional): Evaluate every ``evaluate_every_n_epochs`` epochs.
    """
    set_basic_logging_config()
    _set_hparams()
    add_config(hparams)

    checkpoint = (
        Checkpoint.most_recent(run_root / '**/*.pt')
        if checkpoint_path == '' else Checkpoint.from_path(checkpoint_path))

    step = 0
    directory = Path(run_root) / str(time.strftime('%m_%d', time.localtime())) / run_name
    if checkpoint is not None:
        step = checkpoint.step
        directory = checkpoint.directory.parent

    with TrainingContextManager(root_directory=directory, tensorboard_step=step) as context:
        trainer_kwargs = {}
        if checkpoint is not None:
            logger.info('Loaded checkpoint %s', checkpoint.path)

            if reset_optimizer:
                logger.info('Ignoring checkpoint optimizer.')
                checkpoint.optimizer = None

            trainer_kwargs = {
                'model': checkpoint.model,
                'optimizer': checkpoint.optimizer,
                'epoch': checkpoint.epoch,
                'step': checkpoint.step,
                'anomaly_detector': checkpoint.anomaly_detector
            }

        log_config()
        train, dev = _get_dataset()(
            spectrogram_model_checkpoint_path=spectrogram_model_checkpoint_path)
        trainer = Trainer(context.device, train, dev, context.train_tensorboard,
                          context.dev_tensorboard, **trainer_kwargs)

        # Training Loop
        with True:
            is_trial_run = trainer.epoch == 0 or (checkpoint is not None and
                                                  trainer.epoch == checkpoint['epoch'])
            trainer.run_epoch(train=True, trial_run=is_trial_run)
            if trainer.epoch % evaluate_every_n_epochs == 0:
                Checkpoint(
                    directory=context.checkpoints_directory,
                    model=trainer.model,
                    optimizer=trainer.optimizer,
                    epoch=trainer.epoch,
                    step=trainer.step,
                    anomaly_detector=trainer.anomaly_detector).save()
                trainer.run_epoch(train=False, trial_run=is_trial_run)
            trainer.epoch += 1

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
        '-n', '--name', type=str, default='auto_max_grad_norm', help='Experiment name.')
    parser.add_argument(
        '-r', '--reset_optimizer', action='store_true', default=False, help='Reset optimizer.')
    args, unknown_args = parser.parse_known_args()
    hparams = parse_hparam_args(unknown_args)
    main(
        run_name=args.name,
        checkpoint_path=args.checkpoint,
        spectrogram_model_checkpoint_path=args.spectrogram_model_checkpoint,
        reset_optimizer=args.reset_optimizer,
        hparams=hparams)

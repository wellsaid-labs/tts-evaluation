import argparse
import logging
import random
import os
import glob
import time

from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torchnlp.utils import pad_batch
from tqdm import tqdm

import torch

from src.bin.signal_model._data_iterator import DataIterator
from src.bin.signal_model._utils import load_checkpoint
from src.bin.signal_model._utils import load_data
from src.bin.signal_model._utils import save_checkpoint
from src.bin.signal_model._utils import set_hparams
from src.optimizer import Optimizer
from src.signal_model import WaveRNN
from src.utils import combine_signal
from src.utils import get_total_parameters
from src.utils import parse_hparam_args
from src.utils.configurable import add_config
from src.utils.configurable import configurable
from src.utils.configurable import log_config
from src.utils.experiment_context_manager import ExperimentContextManager

logger = logging.getLogger(__name__)


class Trainer():  # pragma: no cover
    """ Trainer that manages Tacotron training (i.e. running epochs, tensorboard, logging).

    Args:
        device (torch.device): Device to train on.
        train_dataset (iterable): Train dataset used to optimize the model.
        dev_dataset (iterable): Dev dataset used to evaluate.
        train_tensorboard (tensorboardX.SummaryWriter): Writer for train events.
        dev_tensorboard (tensorboardX.SummaryWriter): Writer for dev events.
        sample_rate (int, optional): Sample rate of the audio data.
        train_batch_size (int, optional): Batch size used for training.
        dev_batch_size (int, optional): Batch size used for evaluation.
        model (torch.nn.Module, optional): Model to train and evaluate.
        step (int, optional): Starting step, useful warm starts (i.e. checkpoints).
        epoch (int, optional): Starting epoch, useful warm starts (i.e. checkpoints).
        step_unit (str, optional): Unit to measuer steps in, either: ['batches', 'seconds'].
        criterion (callable): Loss function used to score signal predictions.
        optimizer (torch.optim.Optimizer): Optimizer used for gradient descent.
        num_workers (int, optional): Number of workers for data loading.
    """

    STEP_UNIT_SECONDS = 'seconds'
    STEP_UNIT_BATCHES = 'batches'

    @configurable
    def __init__(self,
                 device,
                 train_dataset,
                 dev_dataset,
                 train_tensorboard,
                 dev_tensorboard,
                 sample_rate=24000,
                 train_batch_size=32,
                 dev_batch_size=128,
                 model=WaveRNN,
                 step=0,
                 epoch=0,
                 step_unit='seconds',
                 criterion=CrossEntropyLoss,
                 optimizer=Adam,
                 num_workers=0):
        assert self.step_unit in [self.STEP_UNIT_BATCHES,
                                  self.STEP_UNIT_SECONDS], 'Picked invalid step unit.'

        # Allow for ``class`` or a class instance
        self.model = model if isinstance(model, torch.nn.Module) else model()
        self.model.to(device)

        self.optimizer = optimizer if isinstance(optimizer, Optimizer) else Optimizer(
            optimizer(params=filter(lambda p: p.requires_grad, self.model.parameters())))
        self.optimizer.to(device)

        self.criterion = criterion(reduce=False).to(self.device)

        self.dev_tensorboard = dev_tensorboard
        self.train_tensorboard = train_tensorboard
        self.device = device
        self.step = step
        self.epoch = epoch
        self.step_unit = step_unit
        self.train_batch_size = train_batch_size
        self.dev_batch_size = dev_batch_size
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.num_workers = num_workers
        self.sample_rate = sample_rate
        self.random = random.Random(123)  # Ensure the same samples are sampled

        logger.info('Training on %d GPUs', torch.cuda.device_count())
        logger.info('Step (%s): %d', self.step_unit, self.step)
        logger.info('Epoch: %d', self.epoch)
        logger.info('Number of Training Rows: %d', len(self.train_dataset))
        logger.info('Number of Dev Rows: %d', len(self.dev_dataset))
        logger.info('Train Batch Size: %d', train_batch_size)
        logger.info('Dev Batch Size: %d', dev_batch_size)
        logger.info('Number of data loading workers: %d', num_workers)
        logger.info('Total Parameters: %d', get_total_parameters(self.model))
        logger.info('Model:\n%s' % self.model)

    def run_epoch(self, train=False, trial_run=False):
        """ Iterate over a dataset with ``self.model``, computing the loss function every iteration.

        Args:
            train (bool): If ``True``, the batch will store gradients.
            trial_run (bool): If ``True``, then runs only 1 batch.
        """
        label = 'TRAIN' if train else 'DEV'
        logger.info('[%s] Running Epoch %d, Step %d', label, self.epoch, self.step)
        if trial_run:
            logger.info('[%s] Trial run with one batch.', label)

        self.tensorboard = self.train_tensorboard if train else self.dev_tensorboard

        # Epoch Average Loss Metrics
        total_coarse_loss, total_fine_loss, total_signal_predictions = 0.0, 0.0, 0

        if self.step_unit == self.STEP_UNIT_SECONDS:
            start = time.time()

        # Setup iterator and metrics
        data_iterator = DataIterator(
            self.device,
            self.train_dataset if train else self.dev_dataset,
            self.train_batch_size if train else self.dev_batch_size,
            trial_run=trial_run,
            num_workers=self.num_workers,
            random=self.random)
        data_iterator = tqdm(data_iterator, desc=label)
        for batch in data_iterator:
            draw_sample = not train and self.random.randint(1, len(data_iterator)) == 1
            coarse_loss, fine_loss, num_signal_predictions = self._run_step(
                batch, train=train, sample=draw_sample)
            total_fine_loss += fine_loss * num_signal_predictions
            total_coarse_loss += coarse_loss * num_signal_predictions
            total_signal_predictions += num_signal_predictions

        if train and self.step_unit == self.STEP_UNIT_SECONDS:
            self.step += int(round(time.time() - start))

        epoch_coarse_loss = total_coarse_loss / total_signal_predictions
        epoch_fine_loss = total_fine_loss / total_signal_predictions
        self.tensorboard.add_scalar('coarse/loss/epoch', epoch_coarse_loss, self.step)
        self.tensorboard.add_scalar('fine/loss/epoch', epoch_fine_loss, self.step)

    def _sample_inference(self, batch, max_infer_frames=100):
        """ Run in inference mode without teacher forcing and push results to Tensorboard.

        Args:
            batch (dict): ``dict`` from ``src.bin.signal_model._utils.DataIterator``.
            max_infer_frames (int, optioanl): Maximum number of frames to consider for memory's
                sake.

        Returns: None
        """
        batch_size = len(batch['log_mel_spectrogram'])
        item = self.random.randint(0, batch_size - 1)

        log_mel_spectrogram = batch['log_mel_spectrogram'][item][:max_infer_frames]
        self.tensorboard.add_log_mel_spectrogram('full/spectrogram', log_mel_spectrogram, self.step)

        scale = int(batch['signal'][item].shape[0] / batch['log_mel_spectrogram'][item].shape[0])
        target_signal = batch['signal'][item][:max_infer_frames * scale]
        self.tensorboard.add_audio('full/gold', 'full/gold_waveform', target_signal, self.step)

        torch.set_grad_enabled(False)
        self.model.train(mode=False)

        logger.info('Running inference on %d spectrogram frames...', log_mel_spectrogram.shape[0])
        predicted_coarse, predicted_fine, _ = self.model(log_mel_spectrogram.unsqueeze(0))
        predicted_signal = combine_signal(predicted_coarse.squeeze(0), predicted_fine.squeeze(0))
        self.tensorboard.add_audio('full/prediction', 'full/prediction_waveform', predicted_signal,
                                   self.step)

    def _sample_predicted(self, batch, predicted_coarse, predicted_fine):
        """ Samples examples from a batch and outputs them to tensorboard.

        Args:
            batch (dict): ``dict`` from ``src.bin.signal_model._utils.DataIterator``.
            predicted_coarse (torch.FloatTensor [batch_size, signal_length, bins])
            predicted_fine (torch.FloatTensor [batch_size, signal_length, bins])

        Returns: None
        """
        # Initial values
        batch_size = predicted_coarse.shape[0]
        item = self.random.randint(0, batch_size - 1)  # Random item to sample
        length = batch['slice']['signal_lengths'][item]

        # Sample argmax from a categorical distribution
        # predicted_signal [batch_size, signal_length, bins] → [signal_length]
        predicted_coarse = predicted_coarse.max(dim=2)[1][item, :length]
        predicted_fine = predicted_fine.max(dim=2)[1][item, :length]
        predicted_signal = combine_signal(predicted_coarse, predicted_fine)
        self.tensorboard.add_audio('slice/prediction_aligned', 'slice/prediction_aligned_waveform',
                                   predicted_signal, self.step)

        # gold_signal [batch_size, signal_length] → [signal_length]
        target_signal_coarse = batch['slice']['target_signal_coarse'][item, :length]
        target_signal_fine = batch['slice']['target_signal_fine'][item, :length]
        target_signal = combine_signal(target_signal_coarse, target_signal_fine)
        self.tensorboard.add_audio('slice/gold', 'slice/gold_waveform', target_signal, self.step)

    def _compute_loss(self, batch, predicted_coarse, predicted_fine):
        """ Compute the loss.

        Args:
            batch (dict): ``dict`` from ``src.bin.signal_model._utils.DataIterator``.
            predicted_coarse (torch.LongTensor [batch_size, signal_length, bins]): Predicted
                categorical distribution over ``bins`` categories for the ``coarse`` random
                variable.
            predicted_fine (torch.LongTensor [batch_size, signal_length, bins]): Predicted
                categorical distribution over ``bins`` categories for the ``fine`` random
                variable.

        Returns:
            coarse_loss (torch.Tensor): Scalar loss value for signal top bits.
            fine_loss (torch.Tensor): Scalar loss value for signal bottom bits.
            num_predictions (int): Number of signal predictions made.
        """
        slice_ = batch['slice']
        mask = [predicted_fine.new_full((length,), 1) for length in slice_['signal_lengths']]
        mask, _ = pad_batch(mask, padding_index=0)  # [batch_size, signal_length]
        num_predictions = torch.sum(mask)

        # [batch_size, signal_length, bins] → [batch_size, bins, signal_length]
        predicted_fine = predicted_fine.transpose(1, 2)
        predicted_coarse = predicted_coarse.transpose(1, 2)

        # coarse_loss [batch_size, signal_length]
        coarse_loss = self.criterion(predicted_coarse, slice_['target_signal_coarse'].long())
        coarse_loss = torch.sum(coarse_loss * mask) / num_predictions

        # fine_loss [batch_size, signal_length]
        fine_loss = self.criterion(predicted_fine, slice_['target_signal_fine'].long())
        fine_loss = torch.sum(fine_loss * mask) / num_predictions

        return coarse_loss, fine_loss, num_predictions

    def _run_step(self, batch, train=False, sample=False):
        """ Computes a batch with ``self.model``, optionally taking a step along the gradient.

        Args:
            batch (dict): ``dict`` from ``src.bin.signal_model._utils.DataIterator``.
            train (bool, optional): If ``True``, takes a optimization step.
            sample (bool, optional): If ``True``, draw sample from step.

        Returns:
            coarse_loss (torch.Tensor): Scalar loss value for signal top bits.
            fine_loss (torch.Tensor): Scalar loss value for signal bottom bits.
            num_predictions (int): Number of signal predictions made.
        """
        torch.set_grad_enabled(train)
        self.model.train(mode=train)

        predicted_coarse, predicted_fine, _ = torch.nn.parallel.data_parallel(
            module=self.model,
            inputs=batch['slice']['log_mel_spectrogram'],
            module_kwargs={
                'input_signal': batch['slice']['input_signal'],
                'target_coarse': batch['slice']['target_signal_coarse'].unsqueeze(2)
            },
            dim=0,
            output_device=self.device)

        coarse_loss, fine_loss, num_predictions = self._compute_loss(
            batch=batch, predicted_coarse=predicted_coarse, predicted_fine=predicted_fine)

        if train:
            self.optimizer.zero_grad()
            (coarse_loss + fine_loss).backward()
            # TODO: Consider using a normal distribution over the last epoch to set this value.
            parameter_norm = self.optimizer.step()
            if parameter_norm is not None:
                self.tensorboard.add_scalar('parameter_norm/step', parameter_norm, self.step)

        coarse_loss, fine_loss = coarse_loss.item(), fine_loss.item()
        predicted_coarse, predicted_fine = predicted_coarse.detach(), predicted_fine.detach()

        if train:
            self.tensorboard.add_scalar('coarse/loss/step', coarse_loss, self.step)
            self.tensorboard.add_scalar('fine/loss/step', fine_loss, self.step)
            if train and self.step_unit == self.STEP_UNIT_BATCHES:
                self.step += 1

        if sample:
            self._sample_predicted(batch, predicted_coarse, predicted_fine)
            self._sample_inference(batch)

        return coarse_loss, fine_loss, num_predictions


def main(checkpoint_path=None,
         epochs=1000,
         train_batch_size=2,
         num_workers=0,
         reset_optimizer=False,
         hparams={},
         dev_to_train_ratio=4,
         evaluate_every_n_epochs=5,
         min_time=60 * 15,
         name=None,
         label='signal_model',
         experiments_root='experiments/'):  # pragma: no cover
    """ Main module that trains a the signal model saving checkpoints incrementally.

    TODO: Consider relabeling this to wave_rnn or signal_model/wave_rnn

    Args:
        checkpoint_path (str, optional): Accepts a checkpoint path to load or empty string
            signaling to load the most recent checkpoint in ``experiments_root``.
        epochs (int, optional): Number of epochs to run for.
        train_batch_size (int, optional): Maximum training batch size.
        num_workers (int, optional): Number of workers for data loading.
        reset_optimizer (bool, optional): Given a checkpoint, resets the optimizer and scheduler.
        hparams (dict, optional): Hparams to override default hparams.
        dev_to_train_ratio (int, optional): Due to various memory requirements, set the ratio
            of dev batch size to train batch size.
        evaluate_every_n_epochs (int, optional): Evaluate every ``evaluate_every_n_epochs`` epochs.
        min_time (int, optional): If an experiment is less than ``min_time`` in seconds, then it's
            files are deleted.
        name (str, optional): Experiment name.
        label (str, optional): Label applied to a experiments from this executable.
        experiments_root (str, optional): Top level directory for all experiments.
    """
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.fastest = False

    if checkpoint_path == '':  # Pick the most recent checkpoint
        checkpoints = os.path.join(experiments_root, label, '**/*.pt')
        checkpoints = list(glob.iglob(checkpoints, recursive=True))
        if len(checkpoints) == 0:
            logger.warn('No checkpoints found')
            checkpoint_path = None
        else:
            checkpoint_path = max(list(checkpoints), key=os.path.getctime)

    checkpoint = load_checkpoint(checkpoint_path)
    directory = None if checkpoint is None else checkpoint['experiment_directory']
    step = 0 if checkpoint is None else checkpoint['step']

    with ExperimentContextManager(
            label=label, min_time=min_time, name=name, directory=directory, step=step) as context:

        if checkpoint_path is not None:
            logger.info('Loaded checkpoint %s', checkpoint_path)

        set_hparams()
        add_config(hparams)
        log_config()
        train, dev = load_data()

        # Set up trainer.
        trainer_kwargs = {}
        if checkpoint is not None:
            del checkpoint['experiment_directory']  # Not useful for kwargs
            if reset_optimizer:
                logger.info('Deleting checkpoint optimizer.')
                del checkpoint['optimizer']
            trainer_kwargs.update(checkpoint)

        trainer = Trainer(
            context.device,
            train,
            dev,
            context.train_tensorboard,
            context.dev_tensorboard,
            train_batch_size=train_batch_size,
            dev_batch_size=train_batch_size * dev_to_train_ratio,
            num_workers=num_workers,
            **trainer_kwargs)

        # Training Loop
        for _ in range(epochs):
            is_trial_run = trainer.epoch == 0 or (checkpoint is not None and
                                                  trainer.epoch == checkpoint['epoch'])
            trainer.run_epoch(train=True, trial_run=is_trial_run)
            if trainer.epoch % evaluate_every_n_epochs == 0:
                save_checkpoint(
                    context.checkpoints_directory,
                    model=trainer.model,
                    optimizer=trainer.optimizer,
                    epoch=trainer.epoch,
                    step=trainer.step,
                    experiment_directory=context.directory)
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
        help='Without a value, loads the most recent checkpoint;'
        'otherwise, expects a checkpoint file path.')
    parser.add_argument('-n', '--name', type=str, default=None, help='Experiment name.')
    parser.add_argument(
        '-b',
        '--train_batch_size',
        type=int,
        default=2,
        help='Set the maximum training batch size; this figure depends on the GPU memory')
    parser.add_argument(
        '-w', '--num_workers', type=int, default=0, help='Numer of workers used for data loading')
    parser.add_argument(
        '-r',
        '--reset_optimizer',
        action='store_true',
        default=False,
        help='Reset optimizer and scheduler.')
    args, unknown_args = parser.parse_known_args()
    hparams = parse_hparam_args(unknown_args)
    main(
        name=args.name,
        checkpoint_path=args.checkpoint,
        train_batch_size=args.train_batch_size,
        num_workers=args.num_workers,
        reset_optimizer=args.reset_optimizer,
        hparams=hparams)

import argparse
import collections
import logging
import os
import random
import time

from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from tqdm import tqdm

import torch

from src.bin.signal_model._data_iterator import DataIterator
from src.bin.signal_model._utils import load_checkpoint
from src.bin.signal_model._utils import load_data
from src.bin.signal_model._utils import save_checkpoint
from src.bin.signal_model._utils import set_hparams
from src.optimizer import AutoOptimizer
from src.optimizer import Optimizer
from src.signal_model import WaveRNN
from src.utils import AnomalyDetector
from src.utils import combine_signal
from src.utils import get_total_parameters
from src.utils import load_most_recent_checkpoint
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
        step_unit (str, optional): Unit to measuer steps in, either: ['batches', 'deciseconds'].
        criterion (callable): Loss function used to score signal predictions.
        optimizer (torch.optim.Optimizer): Optimizer used for gradient descent.
        num_workers (int, optional): Number of workers for data loading.
        anomaly_detector (AnomalyDetector, optional): Anomaly detector used to skip batches that
            result in large loss.
        min_rollback (int, optional): Minimum number of epochs to rollback in case of an anomaly.
    """

    STEP_UNIT_DECISECONDS = 'deciseconds'
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
                 step_unit='deciseconds',
                 criterion=CrossEntropyLoss,
                 optimizer=Adam,
                 num_workers=0,
                 anomaly_detector=None,
                 min_rollback=1):
        assert step_unit in [self.STEP_UNIT_BATCHES,
                             self.STEP_UNIT_DECISECONDS], 'Picked invalid step unit.'

        # Allow for ``class`` or a class instance
        self.model = model if isinstance(model, torch.nn.Module) else model()
        self.model.to(device)

        self.optimizer = optimizer if isinstance(optimizer, Optimizer) else AutoOptimizer(
            optimizer(params=filter(lambda p: p.requires_grad, self.model.parameters())))
        self.optimizer.to(device)

        self.anomaly_detector = anomaly_detector if isinstance(
            anomaly_detector, AnomalyDetector) else AnomalyDetector()

        self.criterion = criterion(reduce=False).to(device)

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
        # NOTE: Rollback ``maxlen=min_rollback + 2`` in case the model enters at a degenerate state
        # at the beginning of the epoch; therefore, allowing us to rollback at least min_rollback
        # epoch every time.
        self.rollback = collections.deque([self.model.state_dict()], min_rollback + 2)

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

        self.train_tensorboard.add_text('event/session', 'Starting new session.', self.step)
        self.dev_tensorboard.add_text('event/session', 'Starting new session.', self.step)

    def _maybe_rollback(self, epoch_coarse_loss):
        """ Maybe rollback the model if the loss is too high.

        Args:
            epoch_coarse_loss (float)
        """
        is_anomaly = self.anomaly_detector.step(epoch_coarse_loss)
        if is_anomaly:
            self.tensorboard.add_text(
                'event/anomaly', 'Rolling back, detected a coarse loss anomaly #%d (%f > %f ± %f)' %
                (self.anomaly_detector.anomaly_counter, epoch_coarse_loss,
                 self.anomaly_detector.last_average,
                 self.anomaly_detector.max_deviation), self.step)
            past_model_state = self.rollback[0]
            self.model.load_state_dict(past_model_state)

            # Clear the possibly degenerative states.
            self.rollback.clear()
            self.rollback.append(past_model_state)
        else:
            self.rollback.append(self.model.state_dict())

    def run_epoch(self, train=False, trial_run=False):
        """ Iterate over a dataset with ``self.model``, computing the loss function every iteration.

        Args:
            train (bool, optional): If ``True``, the batch will store gradients.
            trial_run (bool, optional): If ``True``, then runs only 1 batch.
        """
        label = 'TRAIN' if train else 'DEV'
        logger.info('[%s] Running Epoch %d, Step %d', label, self.epoch, self.step)
        if trial_run:
            logger.info('[%s] Trial run with one batch.', label)

        self.tensorboard = self.train_tensorboard if train else self.dev_tensorboard

        # Epoch Average Loss Metrics
        total_coarse_loss, total_fine_loss, total_signal_predictions = 0.0, 0.0, 0

        # Setup iterator and metrics
        data_iterator = DataIterator(
            self.device,
            self.train_dataset if train else self.dev_dataset,
            self.train_batch_size if train else self.dev_batch_size,
            trial_run=trial_run,
            num_workers=self.num_workers,
            random=self.random)
        data_iterator = tqdm(data_iterator, desc=label)
        for i, batch in enumerate(data_iterator):
            if (i == 0 and self.step_unit == self.STEP_UNIT_DECISECONDS):
                start = time.time() * 10

            draw_sample = not train and self.random.randint(1, len(data_iterator)) == 1
            coarse_loss, fine_loss, num_signal_predictions = self._run_step(
                batch, train=train, sample=draw_sample, epoch_start_time=start)
            total_fine_loss += fine_loss * num_signal_predictions
            total_coarse_loss += coarse_loss * num_signal_predictions
            total_signal_predictions += num_signal_predictions

            # NOTE: This is inside the loop to avoid time to deconstruct the ``data_iterator``
            if (i == len(data_iterator) - 1 and train and
                    self.step_unit == self.STEP_UNIT_DECISECONDS):
                self.step += int(round(time.time() * 10 - start))

        # NOTE: This is not a formal epoch. For example, in the Linda Johnson dataset the average
        # clip size is 6.57 seconds while the average example seen by the model is 900 samples
        # or 0.375 seconds. This epoch means that we've taken a random 900 sample clip from the
        # 13,100 clips in the Linda Johnson dataset.
        #
        # Walking through the math, a real epoch for the Linda Joshon dataset would be about:
        #    Number of samples: 2066808000 = (23h * 60 * 60 + 55m * 60 + 17s) * 24000
        #    This epoch sample size: 11790000 = 13,100 * 900
        #    Formal epoch is 175x larger: 175 ~ 2066808000 / 11790000
        #    Number of batches in formal epoch: 35,882 ~ 2066808000 / 64 / 900
        #
        # Find stats on the Linda Johnson dataset here: https://keithito.com/LJ-Speech-Dataset/
        epoch_coarse_loss = (0 if total_signal_predictions == 0 else
                             total_coarse_loss / total_signal_predictions)
        epoch_fine_loss = (0 if total_signal_predictions == 0 else
                           total_fine_loss / total_signal_predictions)
        if not trial_run:
            self.tensorboard.add_scalar('coarse/loss/epoch', epoch_coarse_loss, self.step)
            self.tensorboard.add_scalar('fine/loss/epoch', epoch_fine_loss, self.step)

        if train:
            self._maybe_rollback(epoch_coarse_loss)

    def _sample_inference(self, batch, max_infer_frames=200):
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
        self.tensorboard.add_log_mel_spectrogram('full/spectrogram', log_mel_spectrogram)

        scale = int(batch['signal'][item].shape[0] / batch['log_mel_spectrogram'][item].shape[0])
        target_signal = batch['signal'][item][:max_infer_frames * scale]
        self.tensorboard.add_audio('full/gold', 'full/gold_waveform', target_signal)

        torch.set_grad_enabled(False)
        self.model.train(mode=False)
        # NOTE: Inference is faster on CPU because of the many small operations being run
        self.model.to(torch.device('cpu'))

        logger.info('Running inference on %d spectrogram frames...', log_mel_spectrogram.shape[0])
        predicted_coarse, predicted_fine, _ = self.model.infer(log_mel_spectrogram.unsqueeze(0))
        predicted_signal = combine_signal(predicted_coarse.squeeze(0), predicted_fine.squeeze(0))
        self.tensorboard.add_audio('full/prediction', 'full/prediction_waveform', predicted_signal)

        self.model.to(self.device)

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
                                   predicted_signal)

        # gold_signal [batch_size, signal_length] → [signal_length]
        target_signal_coarse = batch['slice']['target_signal_coarse'][item, :length]
        target_signal_fine = batch['slice']['target_signal_fine'][item, :length]
        target_signal = combine_signal(target_signal_coarse, target_signal_fine)
        self.tensorboard.add_audio('slice/gold', 'slice/gold_waveform', target_signal)

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
        num_predictions = torch.sum(slice_['signal_mask'])

        # [batch_size, signal_length, bins] → [batch_size, bins, signal_length]
        predicted_fine = predicted_fine.transpose(1, 2)
        predicted_coarse = predicted_coarse.transpose(1, 2)

        # coarse_loss [batch_size, signal_length]
        coarse_loss = self.criterion(predicted_coarse, slice_['target_signal_coarse'].long())
        coarse_loss = torch.sum(coarse_loss * slice_['signal_mask']) / num_predictions

        # fine_loss [batch_size, signal_length]
        fine_loss = self.criterion(predicted_fine, slice_['target_signal_fine'].long())
        fine_loss = torch.sum(fine_loss * slice_['signal_mask']) / num_predictions

        return coarse_loss, fine_loss, num_predictions

    def _run_step(self, batch, train=False, sample=False, epoch_start_time=None):
        """ Computes a batch with ``self.model``, optionally taking a step along the gradient.

        Args:
            batch (dict): ``dict`` from ``src.bin.signal_model._utils.DataIterator``.
            train (bool, optional): If ``True``, takes a optimization step.
            sample (bool, optional): If ``True``, draw sample from step.
            epoch_start_time (float, optional): Epoch start time in deciseconds.

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
            output_device=self.device.index)

        coarse_loss, fine_loss, num_predictions = self._compute_loss(
            batch=batch, predicted_coarse=predicted_coarse, predicted_fine=predicted_fine)

        if self.step_unit == self.STEP_UNIT_DECISECONDS and train:
            step = self.step + int(round(time.time() * 10 - epoch_start_time))
        else:
            if self.step_unit == self.STEP_UNIT_BATCHES and train:
                self.step += 1
            step = self.step

        if train:
            self.optimizer.zero_grad()
            (coarse_loss + fine_loss).backward()
            with self.tensorboard.set_step(step):
                self.optimizer.step(tensorboard=self.tensorboard)

        num_predictions = num_predictions.item()
        coarse_loss, fine_loss = coarse_loss.item(), fine_loss.item()
        predicted_coarse, predicted_fine = predicted_coarse.detach(), predicted_fine.detach()

        if train:
            self.tensorboard.add_scalar('coarse/loss/step', coarse_loss, step)
            self.tensorboard.add_scalar('fine/loss/step', fine_loss, step)

        if sample:
            with self.tensorboard.set_step(step):
                self._sample_predicted(batch, predicted_coarse, predicted_fine)
                self._sample_inference(batch)

        return coarse_loss, fine_loss, num_predictions


def main(checkpoint_path=None,
         epochs=10000,
         reset_optimizer=False,
         hparams={},
         evaluate_every_n_epochs=15,
         min_time=60 * 15,
         name=None,
         label='signal_model',
         experiments_root='experiments/'):  # pragma: no cover
    """ Main module that trains a the signal model saving checkpoints incrementally.

    Args:
        checkpoint_path (str, optional): Accepts a checkpoint path to load or empty string
            signaling to load the most recent checkpoint in ``experiments_root``.
        epochs (int, optional): Number of epochs to run for.
        reset_optimizer (bool, optional): Given a checkpoint, resets the optimizer.
        hparams (dict, optional): Hparams to override default hparams.
        evaluate_every_n_epochs (int, optional): Evaluate every ``evaluate_every_n_epochs`` epochs.
        min_time (int, optional): If an experiment is less than ``min_time`` in seconds, then it's
            files are deleted.
        name (str, optional): Experiment name.
        label (str, optional): Label applied to a experiments from this executable.
        experiments_root (str, optional): Top level directory for all experiments.
    """
    if checkpoint_path == '':
        checkpoints = os.path.join(experiments_root, label, '**/*.pt')
        checkpoint, checkpoint_path = load_most_recent_checkpoint(
            checkpoints, load_checkpoint=load_checkpoint)
    else:
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

        trainer = Trainer(context.device, train, dev, context.train_tensorboard,
                          context.dev_tensorboard, **trainer_kwargs)

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
                    experiment_directory=context.directory,
                    anomaly_detector=trainer.anomaly_detector)
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
    parser.add_argument(
        '-n', '--name', type=str, default='auto_max_grad_norm', help='Experiment name.')
    parser.add_argument(
        '-r', '--reset_optimizer', action='store_true', default=False, help='Reset optimizer.')
    args, unknown_args = parser.parse_known_args()
    hparams = parse_hparam_args(unknown_args)
    main(
        name=args.name,
        checkpoint_path=args.checkpoint,
        reset_optimizer=args.reset_optimizer,
        hparams=hparams)

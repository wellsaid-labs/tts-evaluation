import matplotlib
matplotlib.use('Agg')

import argparse
import logging
import random

from torch.nn import NLLLoss
from torch.optim import Adam
from torchnlp.utils import pad_batch
from tqdm import tqdm

import torch
import tensorflow as tf

from src.audio import inverse_mu_law_quantize
from src.bin.signal_model._utils import DataIterator
from src.bin.signal_model._utils import load_checkpoint
from src.bin.signal_model._utils import save_checkpoint
from src.bin.signal_model._utils import set_hparams
from src.bin.signal_model._utils import load_data
from src.optimizer import Optimizer
from src.signal_model import SignalModel
from src.utils import get_total_parameters
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
        train_batch_size (int, optional): Batch size used for training.
        dev_batch_size (int, optional): Batch size used for evaluation.
        model (torch.nn.Module, optional): Model to train and evaluate.
        step (int, optional): Starting step, useful warm starts (i.e. checkpoints).
        epoch (int, optional): Starting epoch, useful warm starts (i.e. checkpoints).
        criterion (callable): Loss function used to score signal predictions.
        optimizer (torch.optim.Optimizer): Optimizer used for gradient descent.
        num_workers (int, optional): Number of workers for data loading.
    """

    def __init__(self,
                 device,
                 train_dataset,
                 dev_dataset,
                 train_tensorboard,
                 dev_tensorboard,
                 sample_rate,
                 train_batch_size=32,
                 dev_batch_size=128,
                 model=SignalModel,
                 step=0,
                 epoch=0,
                 criterion=NLLLoss,
                 optimizer=Adam,
                 num_workers=0,
                 model_state_dict=None,
                 optimizer_state_dict=None):

        # Allow for ``class`` or a class instance
        self.model = model if isinstance(model, torch.nn.Module) else model()
        self.model.to(device)
        if device.type == 'cuda' and torch.cuda.device_count() > 1:
            logger.info('Training on %d GPUs', torch.cuda.device_count())
            self.model = torch.nn.DataParallel(self.model, dim=0, output_device=device)
        if model_state_dict is not None:
            self.model.load_state_dict(model_state_dict)

        self.optimizer = optimizer if isinstance(optimizer, Optimizer) else Optimizer(
            optimizer(params=filter(lambda p: p.requires_grad, self.model.parameters())))
        if optimizer_state_dict is not None:
            self.optimizer.load_state_dict(optimizer_state_dict)

        self.dev_tensorboard = dev_tensorboard
        self.train_tensorboard = train_tensorboard
        self.tensorboard = train_tensorboard  # Default tensorboard is changed per epoch.
        self.device = device
        self.step = step
        self.epoch = epoch
        self.train_batch_size = train_batch_size
        self.dev_batch_size = dev_batch_size
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.num_workers = num_workers
        self.sample_rate = sample_rate

        self.criterion = criterion(reduce=False).to(self.device)

        logger.info('Number of Training Rows: %d', len(self.train_dataset))
        logger.info('Number of Dev Rows: %d', len(self.dev_dataset))
        logger.info('Train Batch Size: %d', train_batch_size)
        logger.info('Dev Batch Size: %d', dev_batch_size)
        logger.info('Number of data loading workers: %d', num_workers)
        logger.info('Total Parameters: %d', get_total_parameters(self.model))
        logger.info('Model:\n%s' % self.model)

    def _compute_loss(self, gold_signal, gold_signal_lengths, predicted_signal):
        """ Compute the losses for Tacotron.

        Args:
            gold_signal (torch.ShortTensor [batch_size, signal_length])
            gold_signal_lengths (list): Lengths of each signal in the batch.
            predicted_signal (torch.LongTensor [batch_size, mu + 1, signal_length])

        Returns:
            (torch.Tensor) scalar loss values
        """
        mask = [torch.FloatTensor(length).fill_(1) for length in gold_signal_lengths]
        mask, _ = pad_batch(mask, padding_index=0)  # [batch_size, signal_length]
        mask = mask.to(self.device)

        num_predictions = torch.sum(mask)

        # signal_loss [batch_size, signal_length]
        signal_loss = self.criterion(predicted_signal, gold_signal.long())
        signal_loss = torch.sum(signal_loss * mask) / num_predictions

        return signal_loss, num_predictions

    def run_epoch(self, train=False, trial_run=False, teacher_forcing=True):
        """ Iterate over a dataset with ``self.model``, computing the loss function every iteration.

        Args:
            train (bool): If ``True``, the batch will store gradients.
            trial_run (bool): If True, then runs only 1 batch.
            teacher_forcing (bool): Feed ground truth to the model.
        """
        label = 'TRAIN' if train else 'DEV'
        logger.info('[%s] Running Epoch %d, Step %d', label, self.epoch, self.step)
        if trial_run:
            logger.info('[%s] Trial run with one batch.', label)

        # Set mode
        torch.set_grad_enabled(train)
        self.model.train(mode=train)
        self.tensorboard = self.train_tensorboard if train else self.dev_tensorboard

        # Epoch Average Loss Metrics
        total_signal_loss, total_signal_predictions = 0, 0

        # Setup iterator and metrics
        data_iterator = DataIterator(
            self.device,
            self.train_dataset if train else self.dev_dataset,
            self.train_batch_size if train else self.dev_batch_size,
            trial_run=trial_run,
            num_workers=self.num_workers)
        data_iterator = tqdm(data_iterator, desc=label)
        for (source_signal_batch, target_signal_batch, signal_lengths, frames,
             frames_lengths) in data_iterator:
            signal_loss, num_signal_predictions = self._run_step(
                source_signal_batch,
                target_signal_batch,
                signal_lengths,
                frames,
                frames_lengths,
                train=train,
                sample=not train and random.randint(1, len(data_iterator)) == 1)

            total_signal_loss += signal_loss * num_signal_predictions
            total_signal_predictions += num_signal_predictions

        self._add_scalar(['loss', 'signal', 'epoch'], total_signal_loss / total_signal_predictions,
                         self.epoch)

    def _add_scalar(self, path, scalar, step):
        """ Add scalar to tensorboard

        Args:
            path (list): List of tags to use as label.
            scalar (number): Scalar to add to tensorboard.
        """
        path = [s.lower() for s in path]
        self.tensorboard.add_scalar('/'.join(path), scalar, step)

    def _add_audio(self, path, signal, step):
        """ Add audio to tensorboard.

        Args:
            path (list): List of tags to use as label.
            signal (torch.Tensor): Signal to add to tensorboard as audio.
            step (int): Step value to record.
        """
        signal = signal.detach().cpu()
        signal = inverse_mu_law_quantize(signal)
        assert torch.max(signal) <= 1.0 and torch.min(
            signal) >= -1.0, "Should be [-1, 1] it is [%f, %f]" % (torch.max(signal),
                                                                   torch.min(signal))
        self.tensorboard.add_audio('/'.join(path), signal, step, self.sample_rate)

    def _run_step(self,
                  source_signal_batch,
                  target_signal_batch,
                  signal_lengths,
                  frames,
                  frames_lengths,
                  train=False,
                  sample=False):
        """ Computes a batch with ``self.model``, optionally taking a step along the gradient.

        Args:
            source_signal_batch (torch.FloatTensor [batch_size, signal_length]): One timestep
                behind the target signal batch.
            target_signal_batch (torch.FloatTensor [batch_size, signal_length]): Corresponds with
                the predicted signal batch for loss computation.
            signal_lengths (list of int): Lengths of each signal behind padding.
            frames (torch.FloatTensor [batch_size, num_frames, frame_channels]): Spectrogram
                frames used to predict the signal.
            frames_lengths (list of int): Length of each spectrogram before padding.
            train (bool): If ``True``, takes a optimization step.
            sample (bool): If ``True``, samples the current step.

        Returns:
            (torch.Tensor) Loss at every iteration
        """
        predicted_signal = self.model(frames, gold_signal=source_signal_batch)
        signal_loss, num_signal_predictions = self._compute_loss(target_signal_batch,
                                                                 signal_lengths, predicted_signal)

        if train:
            self.optimizer.zero_grad()
            signal_loss.backward()
            parameter_norm = self.optimizer.step()
            if parameter_norm is not None:
                self._add_scalar(['parameter_norm', 'step'], parameter_norm, self.step)
            self.step += 1

        signal_loss = signal_loss.item()

        if train:
            self._add_scalar(['loss', 'signal', 'step'], signal_loss, self.step)

        if sample:
            batch_size = predicted_signal.shape[0]
            item = random.randint(0, batch_size - 1)
            length = signal_lengths[item]

            # gold_frames [batch_size, num_frames, frame_channels] → [batch_size (1), num_frames]
            gold_frames = frames[item, :frames_lengths[item], :].unsqueeze(0)
            # predicted_signal_no_teacher_forcing [batch_size (1), signal_length]
            predicted_signal_no_teacher_forcing = self.model(gold_frames)

            # TODO: Consider sampling with the entire phrase since we cannot do partial conditions

            # predicted_signal [batch_size, mu + 1, signal_length] → [signal_length]
            predicted_signal = predicted_signal.max(dim=1)[1][item, :length]
            # predicted_signal_no_teacher_forcing [batch_size (1), signal_length] → [signal_length]
            predicted_signal_no_teacher_forcing = predicted_signal_no_teacher_forcing[0, :length]
            # gold_signal [batch_size, signal_length] → [signal_length]
            target_signal = target_signal_batch[item, :length]

            # TODO: Save a visualization of the wav
            self._add_audio(['teacher_forcing', 'predicted'], predicted_signal, self.step)
            self._add_audio(['no_teacher_forcing', 'predicted'],
                            predicted_signal_no_teacher_forcing, self.step)
            self._add_audio(['gold'], target_signal, self.step)

        return signal_loss, num_signal_predictions


@configurable
def main(checkpoint=None, epochs=1000, train_batch_size=2, num_workers=0,
         sample_rate=24000):  # pragma: no cover
    """ Main module that trains a the signal model saving checkpoints incrementally.

    Args:
        checkpoint (str, optional): If provided, path to a checkpoint to load.
        epochs (int, optional): Number of epochs to run for.
        train_batch_size (int, optional): Maximum training batch size.
        num_workers (int, optional): Number of workers for data loading.
        source (str, optional): Torch file with the signal dataset including features and signals.
        sample_rate (int, optional): Sample rate of the audio files.
    """
    with ExperimentContextManager(label='signal_model', min_time=60 * 15) as context:
        set_hparams()
        log_config()
        checkpoint = load_checkpoint(checkpoint, context.device)
        train, dev = load_data()

        # Set up trainer.
        trainer_kwargs = {}
        if checkpoint is not None:
            trainer_kwargs = checkpoint

        trainer = Trainer(
            context.device,
            train,
            dev,
            context.train_tensorboard,
            context.dev_tensorboard,
            sample_rate=sample_rate,
            train_batch_size=train_batch_size,
            dev_batch_size=train_batch_size * 4,
            num_workers=num_workers,
            **trainer_kwargs)

        # Training Loop
        for _ in range(epochs):
            is_trial_run = trainer.epoch == 0
            trainer.run_epoch(train=True, trial_run=is_trial_run)
            checkpoint_path = save_checkpoint(
                context.checkpoints_directory,
                model=trainer.model,
                optimizer=trainer.optimizer,
                epoch=trainer.epoch,
                step=trainer.step)
            trainer.run_epoch(train=False, trial_run=is_trial_run)
            trainer.epoch += 1

            print('–' * 100)

    return checkpoint_path


if __name__ == '__main__':  # pragma: no cover
    tf.enable_eager_execution()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--checkpoint", type=str, default=None, help="Load a checkpoint from a path")
    parser.add_argument(
        "-b",
        "--train_batch_size",
        type=int,
        default=2,
        help="Set the maximum training batch size; this figure depends on the GPU memory")
    parser.add_argument(
        "-w", "--num_workers", type=int, default=0, help="Numer of workers used for data loading")
    args = parser.parse_args()
    main(
        checkpoint=args.checkpoint,
        train_batch_size=args.train_batch_size,
        num_workers=args.num_workers)

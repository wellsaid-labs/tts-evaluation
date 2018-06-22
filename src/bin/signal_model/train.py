import argparse
import logging
import random

from torch.nn import NLLLoss
from torch.optim import Adam
from torchnlp.utils import pad_batch
from tqdm import tqdm

import torch

from src.audio import mu_law_decode
from src.bin.signal_model._data_iterator import DataIterator
from src.bin.signal_model._utils import load_checkpoint
from src.bin.signal_model._utils import load_data
from src.bin.signal_model._utils import save_checkpoint
from src.bin.signal_model._utils import set_hparams
from src.optimizer import Optimizer
from src.signal_model import WaveNet
from src.utils import get_total_parameters
from src.utils import parse_hparam_args
from src.utils import plot_log_mel_spectrogram
from src.utils import plot_waveform
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
        train_batch_size (int, optional): Batch size used for training.
        dev_batch_size (int, optional): Batch size used for evaluation.
        model (torch.nn.Module, optional): Model to train and evaluate.
        step (int, optional): Starting step, useful warm starts (i.e. checkpoints).
        epoch (int, optional): Starting epoch, useful warm starts (i.e. checkpoints).
        criterion (callable): Loss function used to score signal predictions.
        optimizer (torch.optim.Optimizer): Optimizer used for gradient descent.
        num_workers (int, optional): Number of workers for data loading.
    """

    @configurable
    def __init__(self,
                 device,
                 train_dataset,
                 dev_dataset,
                 train_tensorboard,
                 dev_tensorboard,
                 sample_rate,
                 train_batch_size=32,
                 dev_batch_size=128,
                 model=WaveNet,
                 step=0,
                 epoch=0,
                 criterion=NLLLoss,
                 optimizer=Adam,
                 num_workers=0):

        # Allow for ``class`` or a class instance
        self.model = model if isinstance(model, torch.nn.Module) else model()
        self.model.to(device)
        self.is_data_parallel = False
        if device.type == 'cuda' and torch.cuda.device_count() > 1:
            logger.info('Training on %d GPUs', torch.cuda.device_count())
            self.is_data_parallel = True
        self.optimizer = optimizer if isinstance(optimizer, Optimizer) else Optimizer(
            optimizer(params=filter(lambda p: p.requires_grad, self.model.parameters())))
        self.dev_tensorboard = dev_tensorboard
        self.train_tensorboard = train_tensorboard
        self.device = device
        self.step = step
        self.epoch = epoch
        self.train_batch_size = train_batch_size
        self.dev_batch_size = dev_batch_size
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.train_dataset.set_receptive_field_size(self.model.receptive_field_size)
        self.dev_dataset.set_receptive_field_size(self.model.receptive_field_size)
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

    def run_epoch(self, train=False, trial_run=False):
        """ Iterate over a dataset with ``self.model``, computing the loss function every iteration.

        Args:
            train (bool): If ``True``, the batch will store gradients.
            trial_run (bool): If True, then runs only 1 batch.
        """
        label = 'TRAIN' if train else 'DEV'
        logger.info('[%s] Running Epoch %d, Step %d', label, self.epoch, self.step)
        if trial_run:
            logger.info('[%s] Trial run with one batch.', label)

        self.tensorboard = self.train_tensorboard if train else self.dev_tensorboard

        # Epoch Average Loss Metrics
        total_signal_loss, total_signal_predictions = 0.0, 0

        # Setup iterator and metrics
        data_iterator = DataIterator(
            self.device,
            self.train_dataset if train else self.dev_dataset,
            self.train_batch_size if train else self.dev_batch_size,
            trial_run=trial_run,
            num_workers=self.num_workers)
        data_iterator = tqdm(data_iterator, desc=label)
        for batch in data_iterator:
            draw_sample = not train and random.randint(1, len(data_iterator)) == 1
            signal_loss, num_signal_predictions = self._run_step(
                batch, train=train, sample=draw_sample)
            total_signal_loss += signal_loss * num_signal_predictions
            total_signal_predictions += num_signal_predictions

        self._add_scalar(['loss', 'epoch'], total_signal_loss / total_signal_predictions, self.step)

    def _add_image(self, path, to_image, step, *data):
        """ Plot data and add image to tensorboard.

        Args:
            path (list): List of tags to use as label.
            to_image (callable): Callable that returns an image given tensor data.
            step (int): Step value to record.
            *tensors (torch.Tensor): Tensor to visualize.
        """
        data = [row.detach().cpu().numpy() if torch.is_tensor(row) else row for row in data]
        self.tensorboard.add_image('/'.join(path), to_image(*data), step)

    def _add_scalar(self, path, scalar, step):
        """ Add scalar to tensorboard

        Args:
            path (list): List of tags to use as label.
            scalar (number): Scalar to add to tensorboard.
        """
        path = [s.lower() for s in path]
        self.tensorboard.add_scalar('/'.join(path), scalar, step)

    def _add_audio(self, path_audio, path_image, signal, step):
        """ Add audio to tensorboard.

        Args:
            path (list): List of tags to use as label.
            signal (torch.Tensor): Signal to add to tensorboard as audio.
            step (int): Step value to record.
        """
        signal = signal.detach().cpu()
        assert torch.max(signal) <= 1.0 and torch.min(
            signal) >= -1.0, "Should be [-1, 1] it is [%f, %f]" % (torch.max(signal),
                                                                   torch.min(signal))
        self.tensorboard.add_audio('/'.join(path_audio), signal, step, self.sample_rate)
        self._add_image(path_image, plot_waveform, step, signal)

    def _infer(self, spectrogram, signal=None, max_infer_frames=300):
        """ Run in inference mode without teacher forcing.

        Args:
            spectrogram (torch.FloatTensor [num_frames, frame_channels]): Spectrogram to run
                inference on.
            signal (torch.FloatTensor [signal_length], optional): Reference signal.
            max_infer_frames (int, optioanl): Maximum number of frames to consider for memory's
                sake.

        Returns:
            predicted_signal (torch.FloatTensor [signal_length]): Predicted signal.
            gold_signal (torch.FloatTensor [signal_length]): Gold signal sliced to aligned with
                predicted signal.
            spectrogram (torch.FloatTensor [num_frames, frame_channels]): Aligned spectrogram to
                predicted signal.
        """
        if signal is not None:
            factor = int(signal.shape[0] / spectrogram.shape[0])
            signal = signal[:max_infer_frames * factor]

        torch.set_grad_enabled(False)
        self.model.train(mode=False)

        # [num_frames, frame_channels] → [batch_size (1), num_frames, frame_channels]
        spectrogram = spectrogram.unsqueeze(0)
        if max_infer_frames is not None:
            spectrogram = spectrogram[:, :max_infer_frames]

        logger.info('Running inference on %d spectrogram frames...', spectrogram.shape[1])
        predicted_signal = self.model(spectrogram).squeeze(0)
        return predicted_signal, signal, spectrogram.squeeze(0)

    def _sample(self, batch, predicted_signal, max_infer_frames=300):
        """ Samples examples from a batch and outputs them to tensorboard

        Args:
            batch (dict): ``dict`` from ``src.bin.signal_model._utils.DataIterator``.
            predicted_signal (torch.FloatTensor [batch_size, signal_channels, signal_length])
            max_infer_frames (int): The number of frames ``nv_wavenet`` has enough memory to
                process to generate realistic samples.

        Returns: None
        """
        batch_size = predicted_signal.shape[0]
        item = random.randint(0, batch_size - 1)
        length = batch['target_signal_lengths'][item]
        # predicted_signal [batch_size, signal_channels, signal_length] → [signal_length]
        predicted_signal = predicted_signal.max(dim=1)[1][item, :length]
        # gold_signal [batch_size, signal_length] → [signal_length]
        target_signal = batch['target_signals'][item, :length]
        self._add_audio(['slice', 'prediction_aligned'], ['slice', 'prediction_aligned_waveform'],
                        mu_law_decode(predicted_signal), self.step)
        self._add_audio(['slice', 'gold'], ['slice', 'gold_waveform'], mu_law_decode(target_signal),
                        self.step)

        # Sample from an inference
        if self.model.queue_kernel_update():
            infered_signal, gold_signal, spectrogram = self._infer(
                spectrogram=batch['spectrograms'][item], signal=batch['signals'][item])
            self._add_audio(['full', 'prediction'], ['full', 'prediction_waveform'],
                            mu_law_decode(infered_signal), self.step)
            self._add_audio(['full', 'gold'], ['full', 'gold_waveform'], gold_signal, self.step)
            self._add_image(['full', 'spectrogram'], plot_log_mel_spectrogram, self.step,
                            spectrogram)

    def _compute_loss(self, gold_signal, gold_signal_lengths, predicted_signal):
        """ Compute the losses for Tacotron.

        Args:
            gold_signal (torch.ShortTensor [batch_size, signal_length])
            gold_signal_lengths (list): Lengths of each signal in the batch.
            predicted_signal (torch.LongTensor [batch_size, signal_channels, signal_length])

        Returns:
            (torch.Tensor) scalar loss values
        """
        mask = [predicted_signal.new_full((length,), 1) for length in gold_signal_lengths]
        mask, _ = pad_batch(mask, padding_index=0)  # [batch_size, signal_length]
        num_predictions = torch.sum(mask)

        # signal_loss [batch_size, signal_length]
        signal_loss = self.criterion(predicted_signal, gold_signal.long())
        signal_loss = torch.sum(signal_loss * mask) / num_predictions

        return signal_loss, num_predictions

    def _run_step(self, batch, train=False, sample=False):
        """ Computes a batch with ``self.model``, optionally taking a step along the gradient.

        Args:
            batch (dict): ``dict`` from ``src.bin.signal_model._utils.DataIterator``.
            train (bool): If ``True``, takes a optimization step.
            sample (bool): If ``True``, samples the current step.
            max_infer_frames (int): The number of frames ``nv_wavenet`` has enough memory to
                process to generate realistic samples.

        Returns:
            (torch.Tensor) Loss at every iteration
        """
        # Set mode
        torch.set_grad_enabled(train)
        self.model.train(mode=train)

        if self.is_data_parallel:
            predicted_signal = torch.nn.parallel.data_parallel(
                module=self.model,
                inputs=batch['frames'],
                module_kwargs={'gold_signal': batch['source_signals']},
                dim=0,
                output_device=self.device)
        else:
            predicted_signal = self.model(batch['frames'], gold_signal=batch['source_signals'])

        # Cut off context
        predicted_signal = predicted_signal[:, :, -batch['target_signals'].shape[1]:]
        signal_loss, num_signal_predictions = self._compute_loss(
            batch['target_signals'], batch['target_signal_lengths'], predicted_signal)

        if train:
            self.optimizer.zero_grad()
            signal_loss.backward()
            parameter_norm = self.optimizer.step()
            if parameter_norm is not None:
                self._add_scalar(['parameter_norm', 'step'], parameter_norm, self.step)
            self.step += 1

        signal_loss = signal_loss.item()
        predicted_signal = predicted_signal.detach()

        if train:
            self._add_scalar(['loss', 'step'], signal_loss, self.step)

        if sample:
            self._sample(batch, predicted_signal)

        return signal_loss, num_signal_predictions


def main(checkpoint=None,
         epochs=1000,
         train_batch_size=2,
         num_workers=0,
         reset_optimizer=False,
         hparams={},
         dev_to_train_ratio=3,
         evaluate_every_n_epochs=5,
         min_time=60 * 15,
         label='signal_model'):  # pragma: no cover
    """ Main module that trains a the signal model saving checkpoints incrementally.

    Args:
        checkpoint (str, optional): If provided, path to a checkpoint to load.
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
        label (str, optional): Label applied to a experiments from this executable.
    """
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.fastest = False

    with ExperimentContextManager(label=label, min_time=min_time) as context:
        set_hparams()
        add_config(hparams)
        log_config()
        checkpoint = load_checkpoint(checkpoint, context.device)
        train, dev = load_data()

        # Set up trainer.
        trainer_kwargs = {}
        if checkpoint is not None:
            if reset_optimizer:
                logger.info('Ignoring loaded optimizer and scheduler.')
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
            is_trial_run = trainer.epoch == 0
            trainer.run_epoch(train=True, trial_run=is_trial_run)
            if trainer.epoch % evaluate_every_n_epochs == 0:
                save_checkpoint(
                    context.checkpoints_directory,
                    model=trainer.model,
                    optimizer=trainer.optimizer,
                    epoch=trainer.epoch,
                    step=trainer.step)
                trainer.run_epoch(train=False, trial_run=is_trial_run)
            trainer.epoch += 1

            print('–' * 100)


if __name__ == '__main__':  # pragma: no cover
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c', '--checkpoint', type=str, default=None, help='Load a checkpoint from a path')
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
    # Assume other arguments correspond to hparams
    hparams = parse_hparam_args(unknown_args)
    # TODO: Add an option to automatically pick the most recent checkpoint to restart;
    # then writing a restart script should be pretty easy.
    main(
        checkpoint=args.checkpoint,
        train_batch_size=args.train_batch_size,
        num_workers=args.num_workers,
        reset_optimizer=args.reset_optimizer,
        hparams=hparams)

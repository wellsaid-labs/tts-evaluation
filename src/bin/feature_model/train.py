import argparse
import logging
import random

from torch.nn import BCELoss
from torch.nn import MSELoss
from torch.optim import Adam
from torch.optim.lr_scheduler import _LRScheduler
from torchnlp.utils import pad_batch
from tqdm import tqdm

import torch

from src.bin.feature_model._data_iterator import DataIterator
from src.bin.feature_model._utils import load_checkpoint
from src.bin.feature_model._utils import load_data
from src.bin.feature_model._utils import save_checkpoint
from src.bin.feature_model._utils import set_hparams
from src.feature_model import FeatureModel
from src.lr_schedulers import DelayedExponentialLR
from src.optimizer import Optimizer
from src.utils import get_total_parameters
from src.utils import plot_attention
from src.utils import plot_log_mel_spectrogram
from src.utils import plot_stop_token
from src.utils.configurable import log_config
from src.utils.experiment_context_manager import ExperimentContextManager

logger = logging.getLogger(__name__)


class Trainer():  # pragma: no cover
    """ Trainer that manages Tacotron training (i.e. running epochs, tensorboard, logging).

    Args:
        device (torch.device): Device to train on.
        train_dataset (iterable): Train dataset used to optimize the model.
        dev_dataset (iterable): Dev dataset used to evaluate.
        vocab_size (int): Size of the input text vocabular used with embeddings.
        train_tensorboard (tensorboardX.SummaryWriter): Writer for train events.
        dev_tensorboard (tensorboardX.SummaryWriter): Writer for dev events.
        train_batch_size (int, optional): Batch size used for training.
        dev_batch_size (int, optional): Batch size used for evaluation.
        model (torch.nn.Module, optional): Model to train and evaluate.
        step (int, optional): Starting step, useful warm starts (i.e. checkpoints).
        epoch (int, optional): Starting epoch, useful warm starts (i.e. checkpoints).
        criterion_frames (callable): Loss function used to score frame predictions.
        criterion_stop_token (callable): Loss function used to score stop token predictions.
        optimizer (torch.optim.Optimizer): Optimizer used for gradient descent.
        scheduler (torch.optim.lr_scheduler): Scheduler used to adjust learning rate.
        num_workers (int, optional): Number of workers for data loading.
    """

    def __init__(self,
                 device,
                 train_dataset,
                 dev_dataset,
                 vocab_size,
                 train_tensorboard,
                 dev_tensorboard,
                 train_batch_size=32,
                 dev_batch_size=128,
                 model=FeatureModel,
                 step=0,
                 epoch=0,
                 criterion_frames=MSELoss,
                 criterion_stop_token=BCELoss,
                 optimizer=Adam,
                 scheduler=DelayedExponentialLR,
                 num_workers=0):

        # Allow for ``class`` or a class instance
        self.model = model if isinstance(model, torch.nn.Module) else model(vocab_size)
        self.model.to(device)
        self.optimizer = optimizer if isinstance(optimizer, Optimizer) else Optimizer(
            optimizer(params=filter(lambda p: p.requires_grad, self.model.parameters())))
        self.scheduler = scheduler if isinstance(scheduler, _LRScheduler) else scheduler(
            self.optimizer.optimizer, last_epoch=step)

        self.dev_tensorboard = dev_tensorboard
        self.train_tensorboard = train_tensorboard
        self.device = device
        self.step = step
        self.epoch = epoch
        self.train_batch_size = train_batch_size
        self.dev_batch_size = dev_batch_size
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.num_workers = num_workers

        self.criterion_frames = criterion_frames(reduce=False).to(self.device)
        self.criterion_stop_token = criterion_stop_token(reduce=False).to(self.device)

        logger.info('Number of Training Rows: %d', len(self.train_dataset))
        logger.info('Number of Dev Rows: %d', len(self.dev_dataset))
        logger.info('Vocab Size: %d', vocab_size)
        logger.info('Train Batch Size: %d', train_batch_size)
        logger.info('Dev Batch Size: %d', dev_batch_size)
        logger.info('Number of data loading workers: %d', num_workers)
        logger.info('Total Parameters: %d', get_total_parameters(self.model))
        logger.info('Model:\n%s' % self.model)

    def _compute_loss(self, batch, predicted_pre_frames, predicted_post_frames,
                      predicted_stop_tokens):
        """ Compute the losses for Tacotron.

        Args:
            batch (dict): ``dict`` from ``src.bin.feature_model._utils.DataIterator``.
            predicted_pre_frames (torch.FloatTensor [num_frames, batch_size, frame_channels]):
                Predicted frames.
            predicted_post_frames (torch.FloatTensor [num_frames, batch_size, frame_channels]):
                Predicted frames with residual.
            predicted_stop_tokens (torch.FloatTensor [num_frames, batch_size]): Predicted stop
                tokens.

        Returns:
            pre_frames_loss (torch.Tensor [scalar])
            post_frames_loss (torch.Tensor [scalar])
            stop_token_loss (torch.Tensor [scalar])
            num_frame_predictions (int): Number of realized frame predictions taking masking into
                account.
            num_stop_token_predictions (int): Number of realized stop token predictions taking
                masking into account.
        """
        # Create masks
        mask = [torch.FloatTensor(length).fill_(1) for length in batch['frame_lengths']]
        mask, _ = pad_batch(mask)  # [batch_size, num_frames]
        mask = mask.to(self.device)
        stop_token_mask = mask.transpose(0, 1)  # [num_frames, batch_size]
        # [num_frames, batch_size] → [num_frames, batch_size, frame_channels]
        frames_mask = stop_token_mask.unsqueeze(2).expand_as(batch['frames'])

        num_frame_predictions = torch.sum(frames_mask)
        num_stop_token_predictions = torch.sum(stop_token_mask)

        # Average loss for pre frames, post frames and stop token loss
        pre_frames_loss = self.criterion_frames(predicted_pre_frames, batch['frames'])
        pre_frames_loss = torch.sum(pre_frames_loss * frames_mask) / num_frame_predictions

        post_frames_loss = self.criterion_frames(predicted_post_frames, batch['frames'])
        post_frames_loss = torch.sum(post_frames_loss * frames_mask) / num_frame_predictions

        stop_token_loss = self.criterion_frames(predicted_stop_tokens, batch['stop_token'])
        stop_token_loss = torch.sum(stop_token_loss * stop_token_mask) / num_stop_token_predictions

        return (pre_frames_loss, post_frames_loss, stop_token_loss, num_frame_predictions,
                num_stop_token_predictions)

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

        # Set mode
        torch.set_grad_enabled(train)
        self.model.train(mode=train)
        self.tensorboard = self.train_tensorboard if train else self.dev_tensorboard

        # Epoch Average Loss Metrics
        total_pre_frames_loss, total_post_frames_loss, total_stop_token_loss = 0, 0, 0
        total_stop_token_predictions, total_frame_predictions = 0, 0

        # Setup iterator and metrics
        data_iterator = tqdm(
            DataIterator(
                self.device,
                self.train_dataset if train else self.dev_dataset,
                self.train_batch_size if train else self.dev_batch_size,
                trial_run=trial_run,
                num_workers=self.num_workers),
            desc=label)
        for batch in data_iterator:
            (pre_frames_loss, post_frames_loss, stop_token_loss, num_frame_predictions,
             num_stop_token_predictions) = self._run_step(
                 batch,
                 train=train,
                 sample=not train and random.randint(1, len(data_iterator)) == 1)

            total_pre_frames_loss += pre_frames_loss * num_frame_predictions
            total_post_frames_loss += post_frames_loss * num_frame_predictions
            total_stop_token_loss += stop_token_loss * num_stop_token_predictions
            total_stop_token_predictions += num_stop_token_predictions
            total_frame_predictions += num_frame_predictions

        self._add_scalar(['pre_frames', 'epoch'], total_pre_frames_loss / total_frame_predictions,
                         self.step)
        self._add_scalar(['post_frames', 'epoch'], total_post_frames_loss / total_frame_predictions,
                         self.step)
        self._add_scalar(['stop_token', 'epoch'],
                         total_stop_token_loss / total_stop_token_predictions, self.step)

    def _add_scalar(self, path, scalar, step):
        """ Add scalar to tensorboard

        Args:
            path (list): List of tags to use as label.
            scalar (number): Scalar to add to tensorboard.
            step (int): Step value to record.
        """
        path = [s.lower() for s in path]
        self.tensorboard.add_scalar('/'.join(path), scalar, step)

    def _add_image(self, path, tensor, to_image, step):
        """ Plot data and add image to tensorboard.

        Args:
            path (list): List of tags to use as label.
            tensor (torch.Tensor): Tensor to visualize.
            to_image (callable): Callable that returns an image given tensor data.
            step (int): Step value to record.
        """
        data = tensor.detach().cpu().numpy()
        self.tensorboard.add_image('/'.join(path), to_image(data), step)

    def _sample(self, batch, predicted_post_frames, predicted_alignments, predicted_stop_tokens):
        """ Samples examples from a batch and outputs them to tensorboard

        Args:
            batch (dict): ``dict`` from ``src.bin.feature_model._utils.DataIterator``.
            predicted_pre_frames (torch.FloatTensor [num_frames, batch_size, frame_channels]):
                Predicted frames.
            predicted_post_frames (torch.FloatTensor [num_frames, batch_size, frame_channels]):
                Predicted frames with residual.
            predicted_stop_tokens (torch.FloatTensor [num_frames, batch_size]): Predicted stop
                tokens.

        Returns: None
        """
        batch_size = predicted_post_frames.shape[1]
        item = random.randint(0, batch_size - 1)
        length = batch['frame_lengths'][item]
        self._add_image(['spectrogram', 'predicted'], predicted_post_frames[:length, item],
                        plot_log_mel_spectrogram, self.step)
        self._add_image(['spectrogram', 'gold'], batch['frames'][:length, item],
                        plot_log_mel_spectrogram, self.step)
        self._add_image(['alignment', 'predicted'], predicted_alignments[:length, item],
                        plot_attention, self.step)
        self._add_image(['stop_token', 'predicted'], predicted_stop_tokens[:length, item],
                        plot_stop_token, self.step)

    def _run_step(self, batch, train=False, sample=False):
        """ Computes a batch with ``self.model``, optionally taking a step along the gradient.

        Args:
            batch (dict): ``dict`` from ``src.bin.feature_model._utils.DataIterator``.
            train (bool): If ``True``, takes a optimization step.
            sample (bool): If ``True``, samples the current step.

        Returns:
            pre_frames_loss (torch.Tensor [scalar])
            post_frames_loss (torch.Tensor [scalar])
            stop_token_loss (torch.Tensor [scalar])
            num_frame_predictions (int): Number of realized frame predictions taking masking into
                account.
            num_stop_token_predictions (int): Number of realized stop token predictions taking
                masking into account.
        """
        (predicted_pre_frames, predicted_post_frames, predicted_stop_tokens,
         predicted_alignments) = self.model(
             batch['text'], ground_truth_frames=batch['frames'])

        (pre_frames_loss, post_frames_loss, stop_token_loss,
         num_frame_predictions, num_stop_token_predictions) = self._compute_loss(
             batch, predicted_pre_frames, predicted_post_frames, predicted_stop_tokens)

        if train:
            self.optimizer.zero_grad()
            (pre_frames_loss + post_frames_loss + stop_token_loss).backward()
            parameter_norm = self.optimizer.step()
            if parameter_norm is not None:
                self._add_scalar(['parameter_norm', 'step'], parameter_norm, self.step)
            self.scheduler.step()
            self.step += 1

        (pre_frames_loss, post_frames_loss, stop_token_loss) = tuple(
            loss.item() for loss in (pre_frames_loss, post_frames_loss, stop_token_loss))

        if train:
            self._add_scalar(['pre_frames', 'step'], pre_frames_loss, self.step)
            self._add_scalar(['post_frames', 'step'], post_frames_loss, self.step)
            self._add_scalar(['stop_token', 'step'], stop_token_loss, self.step)
            for i, lr in enumerate(self.scheduler.get_lr()):
                self._add_scalar(['learning_rate', str(i), 'step'], lr, self.step)

        if sample:
            self._sample(batch, predicted_post_frames, predicted_alignments, predicted_stop_tokens)

        return (pre_frames_loss, post_frames_loss, stop_token_loss, num_frame_predictions,
                num_stop_token_predictions)


def main(checkpoint=None,
         epochs=1000,
         train_batch_size=32,
         num_workers=0,
         reset_optimizer=False,
         dev_to_train_ratio=4,
         min_time=60 * 15,
         label='feature_model'):  # pragma: no cover
    """ Main module that trains a the feature model saving checkpoints incrementally.

    Args:
        checkpoint (str, optional): If provided, path to a checkpoint to load.
        epochs (int, optional): Number of epochs to run for.
        train_batch_size (int, optional): Maximum training batch size.
        num_workers (int, optional): Number of workers for data loading.
        reset_optimizer (bool, optional): Given a checkpoint, resets the optimizer and scheduler.
        dev_to_train_ratio (int, optional): Due to various memory requirements, set the ratio
            of dev batch size to train batch size.
        min_time (int, optional): If an experiment is less than ``min_time`` in seconds, then it's
            files are deleted.
        label (str, optional): Label applied to a experiments from this executable.
    """
    torch.backends.cudnn.enabled = True
    # TODO: Speed test to ensure this is the fastest settings
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.fastest = False

    with ExperimentContextManager(label=label, min_time=min_time) as context:
        set_hparams()
        log_config()
        checkpoint = load_checkpoint(checkpoint, context.device)
        text_encoder = None if checkpoint is None else checkpoint['text_encoder']
        train, dev, text_encoder = load_data(text_encoder=text_encoder)

        # Set up trainer.
        trainer_kwargs = {}
        if checkpoint is not None:
            del checkpoint['text_encoder']
            if reset_optimizer:
                logger.info('Not restoring optimizer and scheduler.')
                del checkpoint['optimizer']
                del checkpoint['scheduler']
            trainer_kwargs = checkpoint
        trainer = Trainer(
            context.device,
            train,
            dev,
            text_encoder.vocab_size,
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
            save_checkpoint(
                context.checkpoints_directory,
                model=trainer.model,
                optimizer=trainer.optimizer,
                scheduler=trainer.scheduler,
                text_encoder=text_encoder,
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
        default=32,
        help='Set the maximum training batch size; this figure depends on the GPU memory')
    parser.add_argument(
        '-w', '--num_workers', type=int, default=0, help='Numer of workers used for data loading')
    parser.add_argument(
        '-r',
        '--reset_optimizer',
        action='store_true',
        default=False,
        help='Reset optimizer and scheduler.')
    args = parser.parse_args()
    main(
        checkpoint=args.checkpoint,
        train_batch_size=args.train_batch_size,
        num_workers=args.num_workers,
        reset_optimizer=args.reset_optimizer,
    )

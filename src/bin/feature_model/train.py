from pathlib import Path

import argparse
import logging
import random

from torch.nn import BCELoss
from torch.nn import MSELoss
from torch.optim import Adam
from tqdm import tqdm

import torch

from src.bin.feature_model._data_iterator import DataIterator
from src.bin.feature_model._utils import load_data
from src.bin.feature_model._utils import set_hparams
from src.feature_model import FeatureModel
from src.optimizer import AutoOptimizer
from src.optimizer import Optimizer
from src.utils import get_total_parameters
from src.utils import load_most_recent_checkpoint
from src.utils import parse_hparam_args
from src.utils import load_checkpoint
from src.utils import save_checkpoint
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
        text_encoder (torchnlp.TextEncoder): Text encoder used to encode and decode the text.
        train_batch_size (int, optional): Batch size used for training.
        dev_batch_size (int, optional): Batch size used for evaluation.
        model (torch.nn.Module, optional): Model to train and evaluate.
        step (int, optional): Starting step, useful warm starts (i.e. checkpoints).
        epoch (int, optional): Starting epoch, useful warm starts (i.e. checkpoints).
        criterion_frames (callable): Loss function used to score frame predictions.
        criterion_stop_token (callable): Loss function used to score stop token predictions.
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
                 text_encoder,
                 train_batch_size=32,
                 dev_batch_size=128,
                 model=FeatureModel,
                 step=0,
                 epoch=0,
                 criterion_frames=MSELoss,
                 criterion_stop_token=BCELoss,
                 optimizer=Adam,
                 num_workers=0):

        # Allow for ``class`` or a class instance
        self.model = model if isinstance(model, torch.nn.Module) else model(text_encoder.vocab_size)
        self.model.to(device)

        self.optimizer = optimizer if isinstance(optimizer, Optimizer) else AutoOptimizer(
            optimizer(params=filter(lambda p: p.requires_grad, self.model.parameters())))
        self.optimizer.to(device)

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
        self.text_encoder = text_encoder

        self.criterion_frames = criterion_frames(reduce=False).to(self.device)
        self.criterion_stop_token = criterion_stop_token(reduce=False).to(self.device)

        logger.info('Training on %d GPUs', torch.cuda.device_count())
        logger.info('Step: %d', self.step)
        logger.info('Epoch: %d', self.epoch)
        logger.info('Number of Training Rows: %d', len(self.train_dataset))
        logger.info('Number of Dev Rows: %d', len(self.dev_dataset))
        logger.info('Vocab Size: %d', text_encoder.vocab_size)
        logger.info('Text Vocab: %s', ', '.join(sorted(self.text_encoder.vocab)))
        logger.info('Train Batch Size: %d', train_batch_size)
        logger.info('Dev Batch Size: %d', dev_batch_size)
        logger.info('Number of data loading workers: %d', num_workers)
        logger.info('Total Parameters: %d', get_total_parameters(self.model))
        logger.info('Model:\n%s', self.model)

        self.train_tensorboard.add_text('event/session', 'Starting new session.', self.step)
        self.dev_tensorboard.add_text('event/session', 'Starting new session.', self.step)

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

        # Set mode
        torch.set_grad_enabled(train)
        self.model.train(mode=train)
        self.tensorboard = self.train_tensorboard if train else self.dev_tensorboard

        # Epoch Average Loss Metrics
        total_pre_frames_loss, total_post_frames_loss, total_stop_token_loss = 0.0, 0.0, 0.0
        total_stop_token_predictions, total_frame_predictions = 0, 0

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

            (pre_frames_loss, post_frames_loss, stop_token_loss, num_frame_predictions,
             num_stop_token_predictions) = self._run_step(
                 batch, train=train, sample=draw_sample)

            total_pre_frames_loss += pre_frames_loss * num_frame_predictions
            total_post_frames_loss += post_frames_loss * num_frame_predictions
            total_stop_token_loss += stop_token_loss * num_stop_token_predictions
            total_stop_token_predictions += num_stop_token_predictions
            total_frame_predictions += num_frame_predictions

        epoch_stop_token_loss = total_stop_token_loss / total_stop_token_predictions
        epoch_pre_frame_loss = total_pre_frames_loss / total_frame_predictions
        epoch_post_frame_loss = total_post_frames_loss / total_frame_predictions
        if not trial_run:
            self.tensorboard.add_scalar('pre_frames/loss/epoch', epoch_pre_frame_loss, self.step)
            self.tensorboard.add_scalar('post_frames/loss/epoch', epoch_post_frame_loss, self.step)
            self.tensorboard.add_scalar('stop_token/loss/epoch', epoch_stop_token_loss, self.step)

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
        num_frame_predictions = torch.sum(batch['frames_mask'])
        num_stop_token_predictions = torch.sum(batch['stop_token_mask'])

        # Average loss for pre frames, post frames and stop token loss
        pre_frames_loss = self.criterion_frames(predicted_pre_frames, batch['frames'])
        pre_frames_loss = torch.sum(pre_frames_loss * batch['frames_mask']) / num_frame_predictions

        post_frames_loss = self.criterion_frames(predicted_post_frames, batch['frames'])
        post_frames_loss = torch.sum(
            post_frames_loss * batch['frames_mask']) / num_frame_predictions

        stop_token_loss = self.criterion_frames(predicted_stop_tokens, batch['stop_token'])
        stop_token_loss = torch.sum(
            stop_token_loss * batch['stop_token_mask']) / num_stop_token_predictions

        return (pre_frames_loss, post_frames_loss, stop_token_loss, num_frame_predictions,
                num_stop_token_predictions)

    def _sample_infered(self, batch, max_infer_frames=1000):
        """ Run in inference mode without teacher forcing and push results to Tensorboard.

        Args:
            batch (dict): ``dict`` from ``src.bin.feature_model._utils.DataIterator``.
            max_infer_frames (int, optioanl): Maximum number of frames to consider for memory's
                sake.

        Returns: None
        """
        batch_size = batch['text'].shape[1]
        item = random.randint(0, batch_size - 1)
        spectrogam_length = batch['frame_lengths'][item]
        text_length = batch['text_lengths'][item]

        text = batch['text'][:text_length, item]
        gold_frames = batch['frames'][:spectrogam_length, item]

        torch.set_grad_enabled(False)
        self.model.train(mode=False)

        logger.info('Running inference...')
        (predicted_pre_frames, predicted_post_frames, predicted_stop_tokens,
         predicted_alignments) = self.model(
             text.unsqueeze(1), max_recursion=max_infer_frames)

        text = self.text_encoder.decode(text)
        predicted_residual = predicted_post_frames - predicted_pre_frames

        with self.tensorboard.set_step(self.step):
            self.tensorboard.add_text('infered/input', text)
            self.tensorboard.add_log_mel_spectrogram('infered/predicted_spectrogram',
                                                     predicted_post_frames[:, 0])
            self.tensorboard.add_log_mel_spectrogram('infered/residual_spectrogram',
                                                     predicted_residual[:, 0])
            self.tensorboard.add_log_mel_spectrogram('infered/gold_spectrogram', gold_frames)
            self.tensorboard.add_attention('infered/alignment', predicted_alignments[:, 0])
            self.tensorboard.add_stop_token('infered/stop_token', predicted_stop_tokens[:, 0])
            self.tensorboard.add_text('infered/input', text)

    def _sample_predicted(self, batch, predicted_pre_frames, predicted_post_frames,
                          predicted_alignments, predicted_stop_tokens):
        """ Samples examples from a batch and outputs them to tensorboard

        Args:
            batch (dict): ``dict`` from ``src.bin.feature_model._utils.DataIterator``.
            predicted_pre_frames (torch.FloatTensor [num_frames, batch_size, frame_channels]):
                Predicted frames without residual.
            predicted_post_frames (torch.FloatTensor [num_frames, batch_size, frame_channels]):
                Predicted frames with residual.
            predicted_alignments (torch.FloatTensor [num_frames, batch_size, num_tokens]):
                Predicted alignments between text and spectrogram.
            predicted_stop_tokens (torch.FloatTensor [num_frames, batch_size]): Predicted stop
                tokens.

        Returns: None
        """
        batch_size = predicted_post_frames.shape[1]
        item = random.randint(0, batch_size - 1)
        spectrogam_length = batch['frame_lengths'][item]
        text_length = batch['text_lengths'][item]

        text = batch['text'][:text_length, item]
        text = self.text_encoder.decode(text)

        predicted_post_frames = predicted_post_frames[:spectrogam_length, item]
        predicted_pre_frames = predicted_pre_frames[:spectrogam_length, item]
        gold_frames = batch['frames'][:spectrogam_length, item]

        predicted_residual = predicted_post_frames - predicted_pre_frames
        predicted_gold_delta = abs(gold_frames - predicted_post_frames)

        predicted_alignments = predicted_alignments[:spectrogam_length, item, :text_length]
        predicted_stop_tokens = predicted_stop_tokens[:spectrogam_length, item]

        with self.tensorboard.set_step(self.step):
            self.tensorboard.add_log_mel_spectrogram('predicted/predicted_spectrogram',
                                                     predicted_post_frames)
            self.tensorboard.add_log_mel_spectrogram('predicted/residual_spectrogram',
                                                     predicted_residual)
            self.tensorboard.add_log_mel_spectrogram('predicted/delta_spectrogram',
                                                     predicted_gold_delta)
            self.tensorboard.add_log_mel_spectrogram('predicted/gold_spectrogram', gold_frames)
            self.tensorboard.add_attention('predicted/alignment', predicted_alignments)
            self.tensorboard.add_stop_token('predicted/stop_token', predicted_stop_tokens)
            self.tensorboard.add_text('predicted/input', text)

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
         predicted_alignments) = self.model(batch['text'], batch['frames'])

        (pre_frames_loss, post_frames_loss, stop_token_loss,
         num_frame_predictions, num_stop_token_predictions) = self._compute_loss(
             batch, predicted_pre_frames, predicted_post_frames, predicted_stop_tokens)

        if train:
            self.optimizer.zero_grad()
            (pre_frames_loss + post_frames_loss + stop_token_loss).backward()
            with self.tensorboard.set_step(self.step):
                self.optimizer.step(tensorboard=self.tensorboard)

        (pre_frames_loss, post_frames_loss, stop_token_loss) = tuple(
            loss.item() for loss in (pre_frames_loss, post_frames_loss, stop_token_loss))

        if train:
            self.tensorboard.add_scalar('pre_frames/loss/step', pre_frames_loss, self.step)
            self.tensorboard.add_scalar('post_frames/loss/step', post_frames_loss, self.step)
            self.tensorboard.add_scalar('stop_token/loss/step', stop_token_loss, self.step)
            self.step += 1

        if sample:
            self._sample_predicted(batch, predicted_pre_frames, predicted_post_frames,
                                   predicted_alignments, predicted_stop_tokens)
            self._sample_infered(batch)

        return (pre_frames_loss, post_frames_loss, stop_token_loss, num_frame_predictions,
                num_stop_token_predictions)


def main(checkpoint_path=None,
         epochs=10000,
         reset_optimizer=False,
         hparams={},
         evaluate_every_n_epochs=1,
         min_time=60 * 15,
         name=None,
         label='feature_model',
         experiments_root='experiments/'):  # pragma: no cover
    """ Main module that trains a the feature model saving checkpoints incrementally.

    Args:
        checkpoint (str, optional): If provided, path to a checkpoint to load.
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
        checkpoints = str(Path(experiments_root) / label / '**/*.pt')
        checkpoint, checkpoint_path = load_most_recent_checkpoint(checkpoints)
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
        text_encoder = None if checkpoint is None else checkpoint['text_encoder']
        train, dev, text_encoder = load_data(text_encoder=text_encoder)

        # Set up trainer.
        trainer_kwargs = {'text_encoder': text_encoder}
        if checkpoint is not None:
            del checkpoint['experiment_directory']
            if reset_optimizer:
                logger.info('Not restoring optimizer.')
                del checkpoint['optimizer']
            trainer_kwargs.update(checkpoint)
        trainer = Trainer(context.device, train, dev, context.train_tensorboard,
                          context.dev_tensorboard, **trainer_kwargs)

        # Training Loop
        for _ in range(epochs):
            is_trial_run = trainer.epoch == 0
            trainer.run_epoch(train=True, trial_run=is_trial_run)
            if trainer.epoch % evaluate_every_n_epochs == 0:
                save_checkpoint(
                    context.checkpoints_directory,
                    model=trainer.model,
                    optimizer=trainer.optimizer,
                    text_encoder=text_encoder,
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
        '-r', '--reset_optimizer', action='store_true', default=False, help='Reset optimizer.')
    args, unknown_args = parser.parse_known_args()
    hparams = parse_hparam_args(unknown_args)
    main(
        name=args.name,
        checkpoint_path=args.checkpoint,
        reset_optimizer=args.reset_optimizer,
        hparams=hparams)

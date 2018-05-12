import matplotlib
matplotlib.use('Agg')

import argparse
import logging
import math
import os
import random

from torch.nn import BCELoss
from torch.nn import MSELoss
from torch.optim import Adam
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from torchnlp.samplers import BucketBatchSampler
from torchnlp.text_encoders import CharacterEncoder
from torchnlp.utils import pad_batch
from tqdm import tqdm

import torch
import tensorflow as tf

tf.enable_eager_execution()

from src.datasets import lj_speech_dataset
from src.utils.experiment_context_manager import ExperimentContextManager
from src.feature_model import FeatureModel
from src.lr_schedulers import DelayedExponentialLR
from src.optimizer import Optimizer
from src.spectrogram import log_mel_spectrogram_to_wav
from src.spectrogram import plot_spectrogram
from src.spectrogram import wav_to_log_mel_spectrogram
from src.utils import get_total_parameters
from src.utils import plot_attention
from src.utils import split_dataset
from src.utils import Loss
from src.utils import plot_loss

logger = logging.getLogger(__name__)


def load_data(context, cache, text_encoder=None, splits=(0.8, 0.2)):
    """ Load the Linda Johnson (LJ) Speech dataset with spectrograms and encoded text.

    Args:
        context (ExperimentContextManager): Context manager for the experiment
        cache (str): Path to cache the processed dataset
        text_encoder (torchnlp.TextEncoder, optional): Text encoder used to encode and decode the
            text.
        splits (tuple): Train and dev splits to use with the dataset

    Returns:
        (list): Linda Johnson (LJ) Speech dataset with ``log_mel_spectrograms`` and ``text``
        (torchnlp.TextEncoder): Text encoder used to encode and decode the text.
    """
    assert len(splits) == 2

    cache = os.path.join(context.root_path, cache)
    if not os.path.isfile(cache):
        data = lj_speech_dataset()
        random.shuffle(data)
        logger.info('Sample Data:\n%s', data[:5])

        if text_encoder is None:
            text_encoder = CharacterEncoder([r['text'] for r in data])
            logger.info('Text encoder vocab size: %d' % text_encoder.vocab_size)

        with tf.device('/gpu:%d' % context.device if context.is_cuda else '/cpu'):
            for row in tqdm(data):
                row['log_mel_spectrograms'] = torch.tensor(wav_to_log_mel_spectrogram(row['wav']))
                row['text'] = torch.tensor(text_encoder.encode(row['text']).data)
                row['stop_token'] = torch.FloatTensor(
                    [0 for _ in range(row['log_mel_spectrograms'].shape[0])])
                row['stop_token'][row['log_mel_spectrograms'].shape[0] - 1] = 1

        train, dev = split_dataset(data, splits)
        logger.info('Number Training Rows: %d', len(train))
        logger.info('Number Dev Rows: %d', len(dev))
        to_save = (train, dev, text_encoder)
        context.save(cache, to_save)
        return to_save

    return context.load(cache)


class DataIterator(object):
    """ Get a batch iterator over the ``dataset``.

    Args:
        context (ExperimentContextManager): Context manager for the experiment
        dataset (list): Dataset to iterate over.
        batch_size (int): Size of the batch for iteration.
        train (bool): If ``True``, the batch will store gradients.
        sort_key (callable): Sort key used to group similar length data used to minimize padding.

    Returns:
        (torch.utils.data.DataLoader) Single-process or multi-process iterators over the dataset.
        Iterator includes variables:
            text_batch (torch.LongTensor [batch_size, num_tokens])
            frames_batch (torch.LongTensor [num_frames, batch_size, frame_channels])
            stop_token_batch (torch.LongTensor [num_frames, batch_size])
    """

    def __init__(self,
                 context,
                 dataset,
                 batch_size,
                 train=True,
                 sort_key=lambda r: r['log_mel_spectrograms'].shape[0],
                 trial_run=False):
        batch_sampler = BucketBatchSampler(dataset, batch_size, False, sort_key=sort_key)
        self.context = context
        self.iterator = DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            collate_fn=DataIterator._collate_fn,
            pin_memory=True,
            num_workers=0)
        self.trial_run = trial_run

    @staticmethod
    def _collate_fn(batch):
        """ List of tensors to a batch variable """
        text_batch, _ = pad_batch([row['text'] for row in batch])
        frame_batch, frame_length_batch = pad_batch([row['log_mel_spectrograms'] for row in batch])
        stop_token_batch, _ = pad_batch([row['stop_token'] for row in batch])
        transpose = lambda b: b.transpose_(0, 1).contiguous()
        return (text_batch, transpose(frame_batch), frame_length_batch, transpose(stop_token_batch))

    def __len__(self):
        return 1 if self.trial_run else len(self.iterator)

    def __iter__(self):
        for batch in self.iterator:
            yield tuple([
                self.context.maybe_cuda(t, non_blocking=True) if torch.is_tesnor(t) else t
                for t in batch
            ])

            if self.trial_run:
                break


class Trainer():
    """ Trainer that manages Tacotron training (i.e. checkpoints, epochs, steps).

    Args:
        context (ExperimentContextManager): Context manager for the experiment
        train_dataset (iterable): Train dataset used to optimize the model.
        dev_dataset (iterable): Dev dataset used to evaluate.
        text_encoder (torchnlp.TextEncoder, optional): Text encoder used to encode and decode the
            text.
        train_batch_size (int, optional): Batch size used for training.
        dev_batch_size (int, optional): Batch size used for evaluation.
        model (torch.nn.Module, optional): Model to train and evaluate.
        step (int, optional): Starting step, useful warm starts (i.e. checkpoints).
        epoch (int, optional): Starting epoch, useful warm starts (i.e. checkpoints).
        criterion_frames (callable): Loss function used to score frame predictions.
        criterion_stop_token (callable): Loss function used to score stop token predictions.
        optimizer (torch.optim.Optimizer): Optimizer used for gradient descent.
        scheduler (torch.optim.lr_scheduler): Scheduler used to adjust learning rate.
    """

    def __init__(self,
                 context,
                 train_dataset,
                 dev_dataset,
                 vocab_size,
                 train_batch_size=36,
                 dev_batch_size=128,
                 model=FeatureModel,
                 step=0,
                 epoch=0,
                 criterion_frames=MSELoss,
                 criterion_stop_token=BCELoss,
                 optimizer=Adam,
                 scheduler=DelayedExponentialLR):

        # Allow for ``class`` or a class instance
        self.model = context.maybe_cuda(
            model if isinstance(model, torch.nn.Module) else model(vocab_size))
        self.optimizer = optimizer if isinstance(optimizer, Optimizer) else Optimizer(
            optimizer(params=filter(lambda p: p.requires_grad, self.model.parameters())))
        self.scheduler = scheduler if isinstance(scheduler, _LRScheduler) else scheduler(
            self.optimizer.optimizer)

        self.context = context
        self.step = step
        self.epoch = epoch
        self.train_batch_size = train_batch_size
        self.dev_batch_size = dev_batch_size
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.loss_pre_frames = context.maybe_cuda(Loss(criterion_frames))
        self.loss_post_frames = context.maybe_cuda(Loss(criterion_frames))
        self.loss_stop_token = context.maybe_cuda(Loss(criterion_stop_token))
        self.best_post_frames_loss = math.inf
        self.best_stop_token_loss = math.inf

        logger.info('Train Batch Size: %d', train_batch_size)
        logger.info('Dev Batch Size: %d', dev_batch_size)
        logger.info('Total Parameters: %d', get_total_parameters(self.model))
        logger.info('Model:\n%s' % self.model)

    def compute_loss(self, gold_frames, gold_frame_lengths, gold_stop_tokens, predicted_pre_frames,
                     predicted_post_frames, predicted_stop_tokens):
        """ Compute the losses for Tacotron.

        Args:
            gold_frames (torch.FloatTensor [num_frames, batch_size, frame_channels]): Ground truth
                frames.
            gold_frame_lengths (list): Lengths of each spectrogram in the batch.
            gold_stop_tokens (torch.FloatTensor [num_frames, batch_size]): Ground truth stop tokens.
            predicted_pre_frames (torch.FloatTensor [num_frames, batch_size, frame_channels]):
                Predicted frames.
            predicted_post_frames (torch.FloatTensor [num_frames, batch_size, frame_channels]):
                Predicted frames with residual
            predicted_stop_tokens (torch.FloatTensor [num_frames, batch_size]): Predicted stop
                tokens.

        Returns:
            (torch.Tensor) scalar loss values
        """
        mask = [torch.FloatTensor(length).fill_(1) for length in gold_frame_lengths]
        mask, _ = pad_batch(mask)  # [batch_size, num_frames]
        mask = mask.transpose(0, 1)  # [num_frames, batch_size]
        pre_frames_loss = self.loss_pre_frames(predicted_pre_frames, gold_frames, mask=mask)
        post_frames_loss = self.loss_post_frames(predicted_post_frames, gold_frames, mask=mask)
        stop_token_loss = self.loss_stop_token(predicted_stop_tokens, gold_stop_tokens, mask=mask)
        return pre_frames_loss, post_frames_loss, stop_token_loss

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
        self.context.epoch(self.epoch)
        torch.set_grad_enabled(train)
        self.model.train(mode=train)

        # Setup iterator and metrics
        dataset = self.train_dataset if train else self.dev_dataset
        max_batch_size = self.train_batch_size if train else self.dev_batch_size
        data_iterator = tqdm(
            DataIterator(self.context, dataset, max_batch_size, train=train, trial_run=trial_run),
            desc=label)
        for (gold_texts, gold_frames, gold_frame_lengths, gold_stop_tokens) in data_iterator:
            pre_frames_loss, post_frames_loss, stop_token_loss = self.run_step(
                gold_texts,
                gold_frames,
                gold_frame_lengths,
                gold_stop_tokens,
                teacher_forcing=teacher_forcing,
                train=train,
                sample=not train and random.randint(1, len(data_iterator)) == 1)

            # Postfix will be displayed on the right of progress bar
            data_iterator.set_postfix(
                pre_frames_loss=self.loss_pre_frames.total / self.loss_pre_frames.num_values,
                post_frames_loss=self.loss_post_frames.total / self.loss_post_frames.num_values,
                stop_token_loss=self.loss_stop_token.total / self.loss_stop_token.num_values)

        epoch_loss_pre_frames = self.loss_pre_frames.epoch()
        epoch_loss_post_frames = self.loss_post_frames.epoch()
        epoch_loss_stop_token = self.loss_stop_token.epoch()

        logger.info('[%s] Pre Frame Loss: %f', label.upper(), epoch_loss_pre_frames)
        logger.info('[%s] Post Frame Loss: %f', label.upper(), epoch_loss_post_frames)
        logger.info('[%s] Stop Token Loss: %f', label.upper(), epoch_loss_stop_token)

        if train:
            self.epoch += 1

        return epoch_loss_pre_frames, epoch_loss_post_frames, epoch_loss_stop_token

    def sample_spectrogram(self, batch, item, filename):
        """ Sample a spectrogram from a batch and save a visualization.

        Args:
            batch (torch.FloatTensor [num_frames, batch_size, frame_channels]): Batch of frames.
            item (int): Item from the batch to sample.
            filename (str): Filename to use for sample without an extension
        """
        _, batch_size, _ = batch.shape
        spectrogram = batch.detach().transpose_(0, 1)[item].cpu().numpy()
        plot_spectrogram(spectrogram, filename + '.png')
        log_mel_spectrogram_to_wav(spectrogram, filename + '.wav')

    def sample_attention(self, batch, item, filename):
        """ Sample an alignment from a batch and save a visualization.

        Args:
            batch (torch.FloatTensor [num_frames, batch_size, num_tokens]): Batch of alignments.
            item (int): Item from the batch to sample.
            filename (str): Filename to use for sample without an extension
        """
        _, batch_size, _ = batch.shape
        alignment = batch.detach().transpose_(0, 1)[item].cpu().numpy()
        plot_attention(alignment, filename + '.png')

    def run_step(self,
                 gold_texts,
                 gold_frames,
                 gold_frame_lengths,
                 gold_stop_tokens,
                 teacher_forcing=True,
                 train=False,
                 sample=False):
        """ Computes a batch with ``self.model``, optionally taking a step along the gradient.

        Args:
            gold_texts (torch.LongTensor [batch_size, num_tokens])
            gold_frames (torch.LongTensor [num_frames, batch_size, frame_channels])
            gold_stop_tokens (torch.LongTensor [num_frames, batch_size])
            teacher_forcing (bool): If ``True``, feed ground truth to the model.
            train (bool): If ``True``, takes a optimization step.
            sample (bool): If ``True``, samples the current step.

        Returns:
            (torch.Tensor) Loss at every iteration
        """
        if teacher_forcing:
            predicted = self.model(gold_texts, ground_truth_frames=gold_frames)
        else:
            predicted = self.model(gold_texts, max_recursion=gold_frames.shape[0])

        (predicted_pre_frames, predicted_post_frames, predicted_stop_tokens,
         predicted_alignments) = predicted

        losses = self.compute_loss(gold_frames, gold_frame_lengths, gold_stop_tokens,
                                   predicted_pre_frames, predicted_post_frames,
                                   predicted_stop_tokens)

        if train:
            sum(losses).backward()
            self.optimizer.step()
            self.scheduler.step()
            self.step += 1
            self.optimizer.zero_grad()

        if sample:
            label = 'train' if train else 'dev'

            def build_filename(base):
                return os.path.join(self.context.epoch_directory,
                                    label + '_' + str(self.step) + '_' + base)

            logger.info('Saving samples in: %s', build_filename('**'))
            _, batch_size, _ = predicted_alignments.shape
            item = random.randint(0, batch_size - 1)
            self.sample_attention(predicted_alignments, item, build_filename('alignment'))
            self.sample_spectrogram(gold_frames, item, build_filename('gold_spectrogram'))
            self.sample_spectrogram(predicted_post_frames, item,
                                    build_filename('predicted_spectrogram'))

        return tuple(t.item() for t in losses)


def load_checkpoint(context, checkpoint=None):
    """ Load a checkpoint.

    Args:
        context (ExperimentContextManager): Context manager for the experiment
        checkpoint (str or None): Path to a checkpoint to load.

    Returns:
        checkpoint (dict or None): Loaded checkpoint or None
    """
    # Load checkpoint
    if checkpoint is not None:
        checkpoint = context.load(os.path.join(context.root_path, checkpoint))
        checkpoint['model'].apply(
            lambda m: m.flatten_parameters() if hasattr(m, 'flatten_parameters') else None)
        # ISSUE: https://github.com/pytorch/pytorch/issues/7255
        checkpoint['scheduler'].optimizer = checkpoint['optimizer'].optimizer
    return checkpoint


def main(checkpoint=None, dataset_cache='data/lj_speech.pt'):
    """ Main module if this file is invoked directly """
    with ExperimentContextManager(label='feature_model') as context:
        checkpoint = load_checkpoint()
        text_encoder = None if checkpoint is None else checkpoint['text_encoder']
        train, dev, text_encoder = load_data(context, dataset_cache, text_encoder=text_encoder)

        # Setup the trainer
        if checkpoint is None:
            trainer = Trainer(context, train, dev, text_encoder.vocab_size)
        else:
            del checkpoint['text_encoder']
            trainer = Trainer(context, train, dev, text_encoder.vocab_size, **checkpoint)

        # Training Loop
        train_losses = []
        dev_losses = []
        while True:
            logger.info('Training...')
            train_losses.append(trainer.run_epoch(train=True, trial_run=(trainer.epoch == 0)))
            # Save Checkpoint
            context.save(
                os.path.join(context.epoch_directory, 'checkpoint.pt'), {
                    'model': trainer.model,
                    'optimizer': trainer.optimizer,
                    'scheduler': trainer.scheduler,
                    'text_encoder': text_encoder,
                    'epoch': trainer.epoch,
                    'step': trainer.step,
                })
            logger.info('Evaluating...')
            dev_losses.append(trainer.run_epoch(train=False, trial_run=(trainer.epoch == 0)))

            # Plot losses
            logger.info('Saving Loss...')
            loss_names = ['Pre Frame Loss', 'Post Frame Loss', 'Stop Token Loss']
            iterator = zip(zip(*train_losses), zip(*dev_losses), loss_names)
            for train_loss, dev_loss, loss_name in iterator:
                filename = loss_name.replace(' ', '_').lower() + '.png'
                directory = os.path.join(context.directory, filename)
                plot_loss(
                    [train_loss, dev_loss], ['Train', 'Dev'], filename=directory, title=loss_name)

            print('–' * 100)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--checkpoint", type=str, default=None, help="load a checkpoint")
    args = parser.parse_args()
    main(checkpoint=args.checkpoint)

from functools import partial

import logging
import random

from torch.nn import BCELoss
from torch.nn import MSELoss
from torch.optim import Adam
from tqdm import tqdm

import torch

from src.audio import griffin_lim
from src.bin.train.spectrogram_model.data_iterator import DataBatchIterator
from src.optimizer import AutoOptimizer
from src.optimizer import Optimizer
from src.spectrogram_model import SpectrogramModel
from src.utils import get_masked_average_norm
from src.utils import get_total_parameters
from src.utils import get_weighted_standard_deviation
from src.hparams import configurable
from src.visualize import CometML
from src.visualize import plot_attention
from src.visualize import plot_spectrogram
from src.visualize import plot_stop_token
from src.hparams import get_config
from src.utils import dict_collapse

logger = logging.getLogger(__name__)


class Trainer():
    """ Trainer that manages Tacotron training (i.e. running epochs, logging, etc.).

    Args:
        device (torch.device): Device to train on.
        train_dataset (iterable): Train dataset used to optimize the model.
        dev_dataset (iterable): Dev dataset used to evaluate.
        text_encoder (torchnlp.TextEncoder): Text encoder used to encode and decode the text.
        comet_ml_project_name (str): Project name, used for grouping experiments.
        comet_ml_experiment_key (str, optioanl): Previous experiment key to restart visualization.
        train_batch_size (int, optional): Batch size used for training.
        dev_batch_size (int, optional): Batch size used for evaluation.
        model (torch.nn.Module, optional): Model to train and evaluate.
        step (int, optional): Starting step, useful warm starts (i.e. checkpoints).
        epoch (int, optional): Starting epoch, useful warm starts (i.e. checkpoints).
        criterion_spectrogram (callable, optional): Loss function used to score frame predictions.
        criterion_stop_token (callable, optional): Loss function used to score stop token
            predictions.
        optimizer (torch.optim.Optimizer, optional): Optimizer used for gradient descent.
    """

    @configurable
    def __init__(self,
                 device,
                 train_dataset,
                 dev_dataset,
                 text_encoder,
                 comet_ml_project_name,
                 comet_ml_experiment_key=None,
                 train_batch_size=32,
                 dev_batch_size=128,
                 model=SpectrogramModel,
                 step=0,
                 epoch=0,
                 criterion_spectrogram=MSELoss,
                 criterion_stop_token=BCELoss,
                 optimizer=Adam):

        # Allow for ``class`` or a class instance
        self.model = model if isinstance(model, torch.nn.Module) else model(text_encoder.vocab_size)
        self.model.to(device)

        self.optimizer = optimizer if isinstance(optimizer, Optimizer) else AutoOptimizer(
            optimizer(params=filter(lambda p: p.requires_grad, self.model.parameters())))
        self.optimizer.to(device)

        self.comet_ml = CometML(
            project_name=comet_ml_project_name, experiment_key=comet_ml_experiment_key)
        self.device = device
        self.step = step
        self.epoch = epoch
        self.train_batch_size = train_batch_size
        self.dev_batch_size = dev_batch_size
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.text_encoder = text_encoder

        self.criterion_spectrogram = criterion_spectrogram(reduction='none').to(self.device)
        self.criterion_stop_token = criterion_stop_token(reduction='none').to(self.device)

        self.comet_ml.set_step(step)
        self.comet_ml.log_current_epoch(epoch)
        self.comet_ml.log_dataset_hash([self.train_dataset, self.dev_dataset])
        self.comet_ml.log_multiple_params(dict_collapse(get_config()))
        self.comet_ml.log_multiple_params({
            'num_parameter': get_total_parameters(self.model),
            'num_gpu': torch.cuda.device_count(),
            'num_training_row': len(self.train_dataset),
            'num_dev_row': len(self.dev_dataset),
            'vocab_size': text_encoder.vocab_size,
            'vocab': sorted(self.text_encoder.vocab),
        })

        logger.info('Training on %d GPUs', torch.cuda.device_count())
        logger.info('Step: %d', self.step)
        logger.info('Epoch: %d', self.epoch)
        logger.info('Train Batch Size: %d', train_batch_size)
        logger.info('Dev Batch Size: %d', dev_batch_size)
        logger.info('Model:\n%s', self.model)

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
        self.comet_ml.context = label.lower()
        self.comet_ml.log_current_epoch(self.epoch)

        # Epoch Average Loss Metrics
        total_pre_spectrogram_loss, total_post_spectrogram_loss = 0.0, 0.0
        total_stop_token_loss = 0.0
        total_attention_norm, total_attention_standard_deviation = 0.0, 0.0
        total_frames, total_spectrogram_values = 0, 0

        # Setup iterator and metrics
        data_iterator = DataBatchIterator(
            self.train_dataset if train else self.dev_dataset,
            self.text_encoder,
            self.train_batch_size if train else self.dev_batch_size,
            self.device,
            trial_run=trial_run)
        data_iterator = tqdm(data_iterator, desc=label, smoothing=0)
        random_batch = random.randint(0, len(data_iterator) - 1)
        for i, batch in enumerate(data_iterator):
            draw_sample = not train and i == random_batch

            (pre_spectrogram_loss, post_spectrogram_loss, stop_token_loss, num_spectrogram_values,
             num_frames, attention_norm, attention_standard_deviation) = self._run_step(
                 batch, train=train, sample=draw_sample)

            total_pre_spectrogram_loss += pre_spectrogram_loss * num_spectrogram_values
            total_post_spectrogram_loss += post_spectrogram_loss * num_spectrogram_values
            total_attention_norm += attention_norm * num_frames
            total_attention_standard_deviation += attention_standard_deviation * num_frames
            total_stop_token_loss += stop_token_loss * num_frames
            total_frames += num_frames
            total_spectrogram_values += num_spectrogram_values

        self.comet_ml.log_epoch_end(self.epoch)

        if not trial_run and train:
            self.epoch += 1

        if trial_run:
            return

        self.comet_ml.log_multiple_metrics({
            'epoch/pre_spectrogram_loss': total_pre_spectrogram_loss / total_spectrogram_values,
            'epoch/post_spectrogram_loss': total_post_spectrogram_loss / total_spectrogram_values,
            'epoch/stop_token_loss': total_stop_token_loss / total_frames,
            'epoch/attention_norm': total_attention_norm / total_frames,
            'epoch/attention_std': total_attention_standard_deviation / total_frames,
        })

    def _compute_loss(self, batch, predicted_pre_spectrogram, predicted_post_spectrogram,
                      predicted_stop_tokens):
        """ Compute the losses for Tacotron.

        Args:
            batch (dict): ``dict`` from ``DataBatchIterator``.
            predicted_pre_spectrogram (torch.FloatTensor [num_frames, batch_size, frame_channels])
            predicted_post_spectrogram (torch.FloatTensor [num_frames, batch_size, frame_channels]):
            predicted_stop_tokens (torch.FloatTensor [num_frames, batch_size])

        Returns:
            pre_spectrogram_loss (torch.Tensor [scalar])
            post_spectrogram_loss (torch.Tensor [scalar])
            stop_token_loss (torch.Tensor [scalar])
            num_frame_predictions (int): Number of frame predictions.
            num_frames (int): Number of frames.
        """
        spectrogram_expanded_mask = batch['spectrogram_expanded_mask']
        spectrogram = batch['spectrogram']

        num_spectrogram_values = torch.sum(spectrogram_expanded_mask)
        num_frames = torch.sum(batch['spectrogram_mask'])

        # Average loss for pre spectrogram, post spectrogram and stop token loss
        pre_spectrogram_loss = self.criterion_spectrogram(predicted_pre_spectrogram, spectrogram)
        pre_spectrogram_loss = torch.sum(
            pre_spectrogram_loss * spectrogram_expanded_mask) / num_spectrogram_values

        post_spectrogram_loss = self.criterion_spectrogram(predicted_post_spectrogram, spectrogram)
        post_spectrogram_loss = torch.sum(
            post_spectrogram_loss * spectrogram_expanded_mask) / num_spectrogram_values

        stop_token_loss = self.criterion_stop_token(predicted_stop_tokens, batch['stop_token'])
        stop_token_loss = (stop_token_loss * batch['spectrogram_mask']).sum() / num_frames

        return (pre_spectrogram_loss, post_spectrogram_loss, stop_token_loss,
                num_spectrogram_values, num_frames)

    def _sample_infered(self, batch, max_infer_frames=1000):
        """ Run in inference mode without teacher forcing and visualizing results.

        Args:
            batch (dict): ``dict`` from ``DataBatchIterator``.
            max_infer_frames (int, optioanl): Maximum number of frames to consider for memory's
                sake.

        Returns: None
        """
        batch_size = batch['text'].shape[1]
        item = random.randint(0, batch_size - 1)
        spectrogam_length = batch['spectrogram_lengths'][item]
        text_length = batch['text_lengths'][item]

        text = batch['text'][:text_length, item]
        gold_spectrogram = batch['spectrogram'][:spectrogam_length, item]

        with torch.no_grad():
            self.model.train(mode=False)

            logger.info('Running inference...')
            (predicted_pre_spectrogram, predicted_post_spectrogram, predicted_stop_tokens,
             predicted_alignments, _) = self.model.infer(text, max_infer_frames)

        text = self.text_encoder.decode(text)
        predicted_residual = predicted_post_spectrogram - predicted_pre_spectrogram

        # [num_frames, num_tokens] → scalar
        attention_norm = get_masked_average_norm(predicted_alignments, dim=1, norm=float('inf'))
        # [num_frames, num_tokens] → scalar
        attention_standard_deviation = get_weighted_standard_deviation(predicted_alignments, dim=1)

        logger.info('Running Griffin-Lim....')
        waveform = griffin_lim(predicted_post_spectrogram.cpu().numpy())

        self.comet_ml.log_multiple_metrics({
            'infered/attention_norm': attention_norm,
            'infered/attention_std': attention_standard_deviation,
        })
        self.comet_ml.log_text_and_audio('infered', text, torch.from_numpy(waveform))
        log_figure = partial(self.comet_ml.log_figure, overwrite=True)
        log_figure('infered/final_spectrogram', plot_spectrogram(predicted_post_spectrogram))
        log_figure('infered/residual_spectrogram', plot_spectrogram(predicted_residual))
        log_figure('infered/gold_spectrogram', plot_spectrogram(gold_spectrogram))
        log_figure('infered/pre_spectrogram', plot_spectrogram(predicted_pre_spectrogram))
        log_figure('infered/alignment', plot_attention(predicted_alignments))
        log_figure('infered/stop_token', plot_stop_token(predicted_stop_tokens))

    def _sample_predicted(self, batch, predicted_pre_spectrogram, predicted_post_spectrogram,
                          predicted_alignments, predicted_stop_tokens):
        """ Samples examples from a batch and visualize them.

        Args:
            batch (dict): ``dict`` from ``DataBatchIterator``.
            predicted_pre_spectrogram (torch.FloatTensor [num_frames, batch_size, frame_channels])
            predicted_post_spectrogram (torch.FloatTensor [num_frames, batch_size, frame_channels])
            predicted_alignments (torch.FloatTensor [num_frames, batch_size, num_tokens])
            predicted_stop_tokens (torch.FloatTensor [num_frames, batch_size])

        Returns: None
        """
        batch_size = predicted_post_spectrogram.shape[1]
        item = random.randint(0, batch_size - 1)
        spectrogam_length = batch['spectrogram_lengths'][item]
        text_length = batch['text_lengths'][item]

        text = batch['text'][:text_length, item]
        text = self.text_encoder.decode(text)

        predicted_post_spectrogram = predicted_post_spectrogram[:spectrogam_length, item]
        predicted_pre_spectrogram = predicted_pre_spectrogram[:spectrogam_length, item]
        gold_spectrogram = batch['spectrogram'][:spectrogam_length, item]

        predicted_residual = predicted_post_spectrogram - predicted_pre_spectrogram
        predicted_gold_delta = abs(gold_spectrogram - predicted_post_spectrogram)

        predicted_alignments = predicted_alignments[:spectrogam_length, item, :text_length]
        predicted_stop_tokens = predicted_stop_tokens[:spectrogam_length, item]

        log_figure = partial(self.comet_ml.log_figure, overwrite=True)
        log_figure('predicted/final_spectrogram', plot_spectrogram(predicted_post_spectrogram))
        log_figure('predicted/residual_spectrogram', plot_spectrogram(predicted_residual))
        log_figure('predicted/delta_spectrogram', plot_spectrogram(predicted_gold_delta))
        log_figure('predicted/gold_spectrogram', plot_spectrogram(gold_spectrogram))
        log_figure('predicted/pre_spectrogram', plot_spectrogram(predicted_pre_spectrogram))
        log_figure('predicted/alignment', plot_attention(predicted_alignments))
        log_figure('predicted/stop_token', plot_stop_token(predicted_stop_tokens))

    def _run_step(self, batch, train=False, sample=False):
        """ Computes a batch with ``self.model``, optionally taking a step along the gradient.

        Args:
            batch (dict): ``dict`` from ``DataBatchIterator``.
            train (bool): If ``True``, takes a optimization step.
            sample (bool): If ``True``, samples the current step.

        Returns:
            pre_spectrogram_loss (torch.Tensor [scalar])
            post_spectrogram_loss (torch.Tensor [scalar])
            stop_token_loss (torch.Tensor [scalar])
            num_frame_predictions (int): Number of realized frame predictions taking masking into
                account.
            num_frames (int): Number of realized stop token predictions taking
                masking into account.
            attention_norm (float): The infinity norm of attention averaged over all alignments.
        """
        (predicted_pre_spectrogram, predicted_post_spectrogram, predicted_stop_tokens,
         predicted_alignments) = self.model(batch['text'], batch['spectrogram'])

        (pre_spectrogram_loss, post_spectrogram_loss,
         stop_token_loss, num_frame_predictions, num_frames) = self._compute_loss(
             batch, predicted_pre_spectrogram, predicted_post_spectrogram, predicted_stop_tokens)

        if train:
            self.optimizer.zero_grad()
            (pre_spectrogram_loss + post_spectrogram_loss + stop_token_loss).backward()
            self.optimizer.step(remote_visualizer=self.comet_ml)

        (pre_spectrogram_loss, post_spectrogram_loss, stop_token_loss) = tuple(
            loss.item() for loss in (pre_spectrogram_loss, post_spectrogram_loss, stop_token_loss))

        # [num_frames, batch_size, num_tokens] → scalar
        attention_norm = get_masked_average_norm(
            predicted_alignments.detach(), dim=2, mask=batch['spectrogram_mask'], norm=float('inf'))
        # [num_frames, batch_size, num_tokens] → scalar
        attention_standard_deviation = get_weighted_standard_deviation(
            predicted_alignments.detach(), dim=2, mask=batch['spectrogram_mask'])

        if train:
            self.comet_ml.log_multiple_metrics({
                'step/pre_spectrogram_loss': pre_spectrogram_loss,
                'step/post_spectrogram_loss': post_spectrogram_loss,
                'step/stop_token_loss': stop_token_loss,
                'step/attention_norm': attention_norm,
                'step/attention_std': attention_standard_deviation,
            })
            self.step += 1
            self.comet_ml.set_step(self.step)

        if sample:
            self._sample_predicted(batch, predicted_pre_spectrogram, predicted_post_spectrogram,
                                   predicted_alignments, predicted_stop_tokens)
            self._sample_infered(batch)

        return (pre_spectrogram_loss, post_spectrogram_loss, stop_token_loss, num_frame_predictions,
                num_frames, attention_norm, attention_standard_deviation)

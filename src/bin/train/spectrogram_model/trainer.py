import logging
import math
import random

from torch.nn import BCEWithLogitsLoss
from torch.nn import MSELoss
from torch.optim import Adam
from torchnlp.text_encoders import CharacterEncoder
from torchnlp.text_encoders import IdentityEncoder

import torch

from src.audio import griffin_lim
from src.bin.train.spectrogram_model.data_loader import DataLoader
from src.datasets import compute_spectrograms
from src.hparams import configurable
from src.hparams import get_config
from src.optimizer import AutoOptimizer
from src.optimizer import Optimizer
from src.spectrogram_model import SpectrogramModel
from src.utils import dict_collapse
from src.utils import evaluate
from src.utils import get_average_norm
from src.utils import get_total_parameters
from src.utils import get_weighted_stdev
from src.utils import lengths_to_mask
from src.visualize import AccumulatedMetrics
from src.visualize import CometML
from src.visualize import plot_attention
from src.visualize import plot_spectrogram
from src.visualize import plot_stop_token

import src.distributed

logger = logging.getLogger(__name__)


class Trainer():
    """ Trainer that manages Tacotron training (i.e. running epochs, logging, etc.).

    Args:
        device (torch.device): Device to train on.
        train_dataset (iterable of TextSpeechRow): Train dataset used to optimize the model.
        dev_dataset (iterable of TextSpeechRow): Dev dataset used to evaluate.
        comet_ml_project_name (str): Project name, used for grouping experiments.
        comet_ml_experiment_key (str, optioanl): Previous experiment key to restart visualization.
        text_encoder (torchnlp.TextEncoder): Text encoder used to encode and decode the text.
        speaker_encoder (torchnlp.TextEncoder): Text encoder used to encode the speaker label.
        train_batch_size (int, optional): Batch size used for training.
        dev_batch_size (int, optional): Batch size used for evaluation.
        model (torch.nn.Module, optional): Model to train and evaluate.
        step (int, optional): Starting step, useful warm starts (i.e. checkpoints).
        epoch (int, optional): Starting epoch, useful warm starts (i.e. checkpoints).
        criterion_spectrogram (callable, optional): Loss function used to score frame predictions.
        criterion_stop_token (callable, optional): Loss function used to score stop
            token predictions.
        optimizer (torch.optim.Optimizer, optional): Optimizer used for gradient descent.
        use_tqdm (bool, optional): Use TQDM to track epoch progress.
    """

    @configurable
    def __init__(self,
                 device,
                 train_dataset,
                 dev_dataset,
                 comet_ml_project_name,
                 comet_ml_experiment_key=None,
                 text_encoder=None,
                 speaker_encoder=None,
                 train_batch_size=32,
                 dev_batch_size=128,
                 model=SpectrogramModel,
                 step=0,
                 epoch=0,
                 criterion_spectrogram=MSELoss,
                 criterion_stop_token=BCEWithLogitsLoss,
                 optimizer=Adam,
                 use_tqdm=False):

        self.train_dataset = compute_spectrograms(train_dataset)
        self.dev_dataset = compute_spectrograms(dev_dataset)

        self.text_encoder = (
            CharacterEncoder([r.text for r in self.train_dataset])
            if text_encoder is None else text_encoder)
        self.speaker_encoder = (
            IdentityEncoder([r.speaker for r in self.train_dataset])
            if speaker_encoder is None else speaker_encoder)

        # Allow for ``class`` or a class instance
        self.model = model if isinstance(model, torch.nn.Module) else model(
            self.text_encoder.vocab_size, self.speaker_encoder.vocab_size)
        self.model.to(device)
        if src.distributed.in_use():
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model, device_ids=[device.index], output_device=device.index, dim=1)

        self.optimizer = optimizer if isinstance(optimizer, Optimizer) else AutoOptimizer(
            optimizer(params=filter(lambda p: p.requires_grad, self.model.parameters())))
        self.optimizer.to(device)

        self.accumulated_metrics = AccumulatedMetrics()

        self.device = device
        self.step = step
        self.epoch = epoch
        self.train_batch_size = train_batch_size
        self.dev_batch_size = dev_batch_size
        self.use_tqdm = use_tqdm

        self.criterion_spectrogram = criterion_spectrogram(reduction='none').to(self.device)
        self.criterion_stop_token = criterion_stop_token(reduction='none').to(self.device)

        self.comet_ml = CometML(
            project_name=comet_ml_project_name,
            experiment_key=comet_ml_experiment_key,
            disabled=src.distributed.in_use() and not src.distributed.is_master())
        self.comet_ml.set_step(step)
        self.comet_ml.log_current_epoch(epoch)
        self.comet_ml.log_dataset_hash([self.train_dataset, self.dev_dataset])
        self.comet_ml.log_parameters(dict_collapse(get_config()))
        self.comet_ml.set_model_graph(str(self.model))
        self.comet_ml.log_parameters({
            'num_parameter': get_total_parameters(self.model),
            'num_training_row': len(self.train_dataset),
            'num_dev_row': len(self.dev_dataset),
            'vocab_size': self.text_encoder.vocab_size,
            'vocab': sorted(self.text_encoder.vocab),
            'num_speakers': self.speaker_encoder.vocab_size,
            'speakers': sorted([str(v) for v in self.speaker_encoder.vocab]),
        })

        logger.info('Training on %d GPUs', torch.cuda.device_count())
        logger.info('Step: %d', self.step)
        logger.info('Epoch: %d', self.epoch)
        logger.info('Train Batch Size: %d', train_batch_size)
        logger.info('Dev Batch Size: %d', dev_batch_size)
        logger.info('Model:\n%s', self.model)
        logger.info('Is Comet ML disabled? %s', 'True' if self.comet_ml.disabled else 'False')

    def run_epoch(self, train=False, trial_run=False, infer=False):
        """ Iterate over a dataset with ``self.model``, computing the loss function every iteration.

        Args:
            train (bool): If ``True``, the batch will store gradients.
            trial_run (bool): If ``True``, then runs only 1 batch.
            infer (bool): If ``True``, this computes attention metrics on the dev set.
        """
        if infer and train:
            raise ValueError('Train and infer are mutually exclusive.')

        if train:
            label = 'TRAIN'
        elif not train and infer:
            label = 'INFER_DEV'
        elif not train:
            label = 'ALIGNED_DEV'

        logger.info('[%s] Running Epoch %d, Step %d', label, self.epoch, self.step)
        if trial_run:
            logger.info('[%s] Trial run with one batch.', label)

        # Set mode(s)
        self.model.train(mode=train)
        self.comet_ml.set_context(label.lower())
        if not trial_run:
            self.comet_ml.log_current_epoch(self.epoch)

        # Setup iterator and metrics
        dataset = self.train_dataset if train else self.dev_dataset
        data_loader = DataLoader(
            data=dataset,
            batch_size=self.train_batch_size if train else self.dev_batch_size,
            device=self.device,
            text_encoder=self.text_encoder,
            speaker_encoder=self.speaker_encoder,
            trial_run=trial_run,
            use_tqdm=self.use_tqdm)

        # Run epoch
        random_batch = random.randint(0, len(data_loader) - 1)
        for i, batch in enumerate(data_loader):
            with torch.set_grad_enabled(train):
                if infer:
                    predictions = self.model(batch.text[0], batch.speaker[0])
                else:
                    predictions = self.model(batch.text[0], batch.speaker[0], batch.spectrogram[0])
                    self._do_loss_and_maybe_backwards(batch, predictions, do_backwards=train)
                predictions = [p.detach() if torch.is_tensor(p) else p for p in predictions]
                spectrogram_lengths = predictions[-1] if infer else batch.spectrogram[1]
                self._add_attention_metrics(predictions[3], spectrogram_lengths)

            if not train and not infer and i == random_batch:
                self._visualize_predicted(batch, predictions)

            self.accumulated_metrics.log_step_end(
                lambda k, v: self.comet_ml.log_metric('step/' + k, v) if train else None)
            if train:
                self.step += 1
                self.comet_ml.set_step(self.step)

        # Log epoch metrics
        if not trial_run:
            self.comet_ml.log_epoch_end(self.epoch)
            self.accumulated_metrics.log_epoch_end(
                lambda k, v: self.comet_ml.log_metric('epoch/' + k, v))
            if train:
                self.epoch += 1

    def _do_loss_and_maybe_backwards(self, batch, predictions, do_backwards):
        """ Compute the losses and maybe do backwards.

        Args:
            batch (SpectrogramModelTrainingRow)
            predictions (any): Return value from ``self.model.forwards``.
            do_backwards (bool): If ``True`` backward propogate the loss.
        """
        (predicted_pre_spectrogram, predicted_post_spectrogram, predicted_stop_tokens,
         predicted_alignments) = predictions
        spectrogram = batch.spectrogram[0]

        expanded_mask = batch.spectrogram_expanded_mask[0]
        pre_spectrogram_loss = self.criterion_spectrogram(predicted_pre_spectrogram, spectrogram)
        pre_spectrogram_loss = pre_spectrogram_loss.masked_select(expanded_mask).mean()

        post_spectrogram_loss = self.criterion_spectrogram(predicted_post_spectrogram, spectrogram)
        post_spectrogram_loss = post_spectrogram_loss.masked_select(expanded_mask).mean()

        mask = batch.spectrogram_mask[0]
        stop_token_loss = self.criterion_stop_token(predicted_stop_tokens, batch.stop_token[0])
        stop_token_loss = stop_token_loss.masked_select(mask).mean()

        if do_backwards:
            self.optimizer.zero_grad()
            (pre_spectrogram_loss + post_spectrogram_loss + stop_token_loss).backward()
            self.optimizer.step(comet_ml=self.comet_ml)

        # Record metrics
        self.accumulated_metrics.add_multiple_metrics({
            'pre_spectrogram_loss': pre_spectrogram_loss,
            'post_spectrogram_loss': post_spectrogram_loss,
        }, expanded_mask.sum())
        self.accumulated_metrics.add_multiple_metrics({
            'stop_token_loss': stop_token_loss
        }, mask.sum())

        return (pre_spectrogram_loss, post_spectrogram_loss, stop_token_loss, expanded_mask.sum(),
                mask.sum())

    def _add_attention_metrics(self, predicted_alignments, lengths):
        """ Compute and report attention metrics """
        # predicted_alignments [num_frames, batch_size, num_tokens]
        mask = lengths_to_mask(lengths, device=predicted_alignments.device).transpose(0, 1)
        kwargs = {'tensor': predicted_alignments.detach(), 'dim': 2, 'mask': mask}
        self.accumulated_metrics.add_multiple_metrics({
            'attention_norm': get_average_norm(norm=math.inf, **kwargs),
            'attention_std': get_weighted_stdev(**kwargs),
        }, kwargs['mask'].sum())

    def visualize_infered(self):
        """ Run in inference mode without teacher forcing and visualizing results.
        """
        if src.distributed.in_use() and not src.distributed.is_master():
            return

        example = random.sample(self.dev_dataset, 1)[0]

        text = self.text_encoder.encode(example.text).to(self.device)
        speaker = self.speaker_encoder.encode(example.speaker).to(self.device)
        model = self.model.module if src.distributed.in_use() else self.model

        with evaluate(model, device=self.device):
            logger.info('Running inference...')
            (predicted_pre_spectrogram, predicted_post_spectrogram, predicted_stop_tokens,
             predicted_alignments, predicted_lengths) = model(text, speaker)

        predicted_residual = predicted_post_spectrogram - predicted_pre_spectrogram

        self._add_attention_metrics(predicted_alignments, predicted_lengths)
        self.comet_ml.log_audio(
            tag='infered',
            text=example.text,
            speaker=example.speaker,
            predicted_audio=griffin_lim(predicted_post_spectrogram.cpu().numpy()),
            gold_audio=example.spectrogram_audio.to_tensor())
        self.comet_ml.log_figures({
            'infered/final_spectrogram': plot_spectrogram(predicted_post_spectrogram),
            'infered/residual_spectrogram': plot_spectrogram(predicted_residual),
            'infered/gold_spectrogram': plot_spectrogram(example.spectrogram.to_tensor()),
            'infered/pre_spectrogram': plot_spectrogram(predicted_pre_spectrogram),
            'infered/alignment': plot_attention(predicted_alignments),
            'infered/stop_token': plot_stop_token(predicted_stop_tokens),
        })

    def _visualize_predicted(self, batch, predictions):
        """ Visualize examples from a batch and visualize them.

        Args:
            batch (SpectrogramModelTrainingRow)
            predictions (any): Return value from ``self.model.forwards``.
        """
        (predicted_pre_spectrogram, predicted_post_spectrogram, predicted_stop_tokens,
         predicted_alignments) = predictions
        batch_size = predicted_post_spectrogram.shape[1]
        item = random.randint(0, batch_size - 1)
        spectrogam_length = batch.spectrogram[1][item]
        text_length = batch.text[1][item]

        predicted_post_spectrogram = predicted_post_spectrogram[:spectrogam_length, item]
        predicted_pre_spectrogram = predicted_pre_spectrogram[:spectrogam_length, item]
        gold_spectrogram = batch.spectrogram[0][:spectrogam_length, item]

        predicted_residual = predicted_post_spectrogram - predicted_pre_spectrogram
        predicted_delta = abs(gold_spectrogram - predicted_post_spectrogram)

        predicted_alignments = predicted_alignments[:spectrogam_length, item, :text_length]
        predicted_stop_tokens = predicted_stop_tokens[:spectrogam_length, item]

        self.comet_ml.log_figures({
            'predicted/final_spectrogram': plot_spectrogram(predicted_post_spectrogram),
            'predicted/residual_spectrogram': plot_spectrogram(predicted_residual),
            'predicted/delta_spectrogram': plot_spectrogram(predicted_delta),
            'predicted/gold_spectrogram': plot_spectrogram(gold_spectrogram),
            'predicted/pre_spectrogram': plot_spectrogram(predicted_pre_spectrogram),
            'predicted/alignment': plot_attention(predicted_alignments),
            'predicted/stop_token': plot_stop_token(predicted_stop_tokens),
        })

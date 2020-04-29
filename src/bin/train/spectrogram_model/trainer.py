import atexit
import itertools
import logging
import math
import random

from hparams import configurable
from hparams import get_config
from hparams import HParam
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
from torchnlp.random import fork_rng
from torchnlp.utils import get_total_parameters
from torchnlp.utils import lengths_to_mask
from torchnlp.utils import tensors_to

import torch

from src.audio import db_to_power
from src.audio import framed_rms_from_power_spectrogram
from src.audio import griffin_lim
from src.audio import power_to_db
from src.bin.train.spectrogram_model.data_loader import DataLoader
from src.optimizers import AutoOptimizer
from src.optimizers import Optimizer
from src.spectrogram_model import InputEncoder
from src.utils import Checkpoint
from src.utils import dict_collapse
from src.utils import DistributedAveragedMetric
from src.utils import evaluate
from src.utils import get_average_norm
from src.utils import get_weighted_stdev
from src.utils import log_runtime
from src.utils import maybe_load_tensor
from src.utils import mean
from src.utils import random_sample
from src.utils import RepeatTimer
from src.visualize import plot_attention
from src.visualize import plot_loss_per_frame
from src.visualize import plot_mel_spectrogram
from src.visualize import plot_stop_token

import src.distributed

logger = logging.getLogger(__name__)

# TODO: Consider re-organizing `Trainer` to be more functional, with each function being stateless
# and rather simply changing state. This more closely fits with the "checkpoint" pattern. Testing
# will be simpler because we can avoid potentially mocking all the objects created during
# `Trainer` instantiation. Finally, it aligns more closely with PyTorch's design.


class Trainer():
    """ Trainer defines a simple interface for training the ``SpectrogramModel``.

    Args:
        device (torch.device): Device to train on.
        train_dataset (iterable of TextSpeechRow): Train dataset used to optimize the model.
        dev_dataset (iterable of TextSpeechRow): Dev dataset used to evaluate the model.
        checkpoints_directory (str or Path): Directory to store checkpoints in.
        comet_ml (Experiment or ExistingExperiment): Object for visualization with comet.
        train_batch_size (int): Batch size used for training.
        dev_batch_size (int): Batch size used for evaluation.
        criterion_spectrogram (callable): Loss function used to score frame predictions.
        criterion_stop_token (callable): Loss function used to score stop
            token predictions.
        optimizer (torch.optim.Optimizer): Optimizer used for gradient descent.
        model (torch.nn.Module): Model to train and evaluate.
        input_encoder (src.spectrogram_model.InputEncoder): Spectrogram model input encoder.
        step (int, optional): Starting step; typically, this parameter is useful when starting from
            a checkpoint.
        epoch (int, optional): Starting epoch; typically, this parameter is useful when starting
            from a checkpoint.
        save_temp_checkpoint_every_n_seconds (int, optional): The number of seconds between
            temporary checkpoint saves.
        dataset_sample_size (int, optional): The number of samples to compute expensive dataset
            statistics.
    """

    TRAIN_LABEL = 'train'
    DEV_INFERRED_LABEL = 'dev_inferred'
    DEV_LABEL = 'dev'

    @configurable
    def __init__(self,
                 device,
                 train_dataset,
                 dev_dataset,
                 checkpoints_directory,
                 comet_ml,
                 train_batch_size=HParam(),
                 dev_batch_size=HParam(),
                 criterion_spectrogram=HParam(),
                 criterion_stop_token=HParam(),
                 optimizer=HParam(),
                 lr_multiplier_schedule=HParam(),
                 model=HParam(),
                 input_encoder=None,
                 step=0,
                 epoch=0,
                 save_temp_checkpoint_every_n_seconds=60 * 10,
                 dataset_sample_size=50):
        self.device = device
        self.step = step
        self.epoch = epoch
        self.checkpoints_directory = checkpoints_directory
        self.dev_dataset = dev_dataset
        self.train_dataset = train_dataset
        self.train_batch_size = train_batch_size
        self.dev_batch_size = dev_batch_size

        # TODO: The `input_encoder` should not have any insight onto the `dev_dataset`. There
        # should be a process for dealing with unknown characters instead.
        corpus = [r.text for r in itertools.chain(self.train_dataset, self.dev_dataset)]
        speakers = [r.speaker for r in itertools.chain(self.train_dataset, self.dev_dataset)]
        self.input_encoder = (
            InputEncoder(corpus, speakers) if input_encoder is None else input_encoder)

        num_tokens = self.input_encoder.text_encoder.vocab_size
        num_speakers = self.input_encoder.speaker_encoder.vocab_size
        # NOTE: Allow for `class` or a class instance.
        self.model = model if isinstance(model, nn.Module) else model(num_tokens, num_speakers)
        self.model.to(device)
        if src.distributed.is_initialized():
            self.model = nn.parallel.DistributedDataParallel(
                self.model, device_ids=[device], output_device=device, dim=1)

        self.optimizer = optimizer if isinstance(optimizer, Optimizer) else AutoOptimizer(
            optimizer(params=filter(lambda p: p.requires_grad, self.model.parameters())))
        self.optimizer.to(device)

        self.scheduler = LambdaLR(
            self.optimizer.optimizer, lr_multiplier_schedule, last_epoch=step - 1)

        self.metrics = {
            'attention_norm': DistributedAveragedMetric(),
            'attention_std': DistributedAveragedMetric(),
            'data_queue_size': DistributedAveragedMetric(),
            'average_relative_speed': DistributedAveragedMetric(),
            'post_spectrogram_loss': DistributedAveragedMetric(),
            'pre_spectrogram_loss': DistributedAveragedMetric(),
            'reached_max_frames': DistributedAveragedMetric(),
            'stop_token_accuracy': DistributedAveragedMetric(),
            'stop_token_loss': DistributedAveragedMetric(),
        }
        self.loudness_metrics = {
            'average_target_loudness': DistributedAveragedMetric(),
            'average_predicted_loudness': DistributedAveragedMetric(),
        }

        self.criterion_spectrogram = criterion_spectrogram(reduction='none').to(self.device)
        self.criterion_stop_token = criterion_stop_token(reduction='none').to(self.device)

        self.comet_ml = comet_ml
        self.comet_ml.set_step(step)
        self.comet_ml.log_current_epoch(epoch)
        self.comet_ml.log_parameters(dict_collapse(get_config()))
        self.comet_ml.set_model_graph(str(self.model))

        self.comet_ml.log_parameters({
            'num_parameter': get_total_parameters(self.model),
            'num_training_row': len(self.train_dataset),
            'num_dev_row': len(self.dev_dataset),
            'vocab_size': self.input_encoder.text_encoder.vocab_size,
            'vocab': sorted(self.input_encoder.text_encoder.vocab),
            'num_speakers': self.input_encoder.speaker_encoder.vocab_size,
            'speakers': sorted([str(v) for v in self.input_encoder.speaker_encoder.vocab]),
            'average_train_text_length': mean(len(r.text) for r in self.train_dataset),
            'average_dev_text_length': mean(len(r.text) for r in self.dev_dataset),
        })
        self.comet_ml.log_parameters({
            # NOTE: The average text and spectrogram length can be used to ensure that two datasets
            # are equivalent between two experiments.
            'average_train_spectrogram_length':
                mean(r.spectrogram.shape[0] for r in self.train_dataset),
            'average_dev_spectrogram_length':
                mean(r.spectrogram.shape[0] for r in self.dev_dataset),
            'expected_average_train_spectrogram_sum':
                self._get_expected_average_spectrogram_sum(self.train_dataset, dataset_sample_size),
            'expected_average_dev_spectrogram_sum':
                self._get_expected_average_spectrogram_sum(self.dev_dataset, dataset_sample_size)
        })

        logger.info('Training on %d GPUs', torch.cuda.device_count())
        logger.info('Step: %d', self.step)
        logger.info('Vocab: %s', sorted(self.input_encoder.text_encoder.vocab))
        logger.info('Epoch: %d', self.epoch)
        logger.info('Train Batch Size: %d', train_batch_size)
        logger.info('Dev Batch Size: %d', dev_batch_size)
        logger.info('Model:\n%s', self.model)
        logger.info('Is Comet ML disabled? %s', 'True' if self.comet_ml.disabled else 'False')

        if src.distributed.is_master():
            self.timer = RepeatTimer(save_temp_checkpoint_every_n_seconds,
                                     self._save_checkpoint_repeat_timer)
            self.timer.daemon = True
            self.timer.start()
            atexit.register(self.save_checkpoint)

    def _save_checkpoint_repeat_timer(self):
        """ Save a checkpoint and delete the last checkpoint saved.
        """
        # NOTE: GCP shutdowns do not trigger `atexit`; therefore, it's useful to always save
        # a temporary checkpoint just in case.
        logger.info('Saving temporary checkpoint...')
        checkpoint_path = self.save_checkpoint()
        if (hasattr(self, '_last_repeat_timer_checkpoint') and
                self._last_repeat_timer_checkpoint is not None and
                self._last_repeat_timer_checkpoint.exists() and checkpoint_path is not None):
            logger.info('Unlinking temporary checkpoint: %s',
                        str(self._last_repeat_timer_checkpoint))
            self._last_repeat_timer_checkpoint.unlink()
        self._last_repeat_timer_checkpoint = checkpoint_path

    @log_runtime
    def _get_expected_average_spectrogram_sum(self, dataset, sample_size):
        """
        Args:
            dataset (iterable of TextSpeechRow)
            sample_size (int)

        Returns:
            (float): Mean of the sum of a sample of spectrograms from `dataset`.
        """
        if src.distributed.is_master():
            with fork_rng(seed=123):
                sample = random_sample(dataset, sample_size)
                return mean(maybe_load_tensor(r.spectrogram).sum().item() for r in sample)
        return None

    @classmethod
    def from_checkpoint(class_, checkpoint, **kwargs):
        """ Instantiate ``Trainer`` from a checkpoint.

        Args:
            checkpoint (Checkpoint): Checkpoint to initiate ``Trainer`` with.
            **kwargs: Additional keyword arguments passed to ``__init__``.

        Returns:
            (Trainer)
        """
        checkpoint_kwargs = {
            'model': checkpoint.model,
            'optimizer': checkpoint.optimizer,
            'epoch': checkpoint.epoch,
            'step': checkpoint.step,
            'input_encoder': checkpoint.input_encoder,
        }
        checkpoint_kwargs.update(kwargs)
        return class_(**checkpoint_kwargs)

    def save_checkpoint(self):
        """ Save a checkpoint.

        Returns:
            (pathlib.Path or None): Path the checkpoint was saved to or None if checkpoint wasn't
                saved.
        """
        if src.distributed.is_master():
            checkpoint = Checkpoint(
                comet_ml_project_name=self.comet_ml.project_name,
                directory=self.checkpoints_directory,
                model=(self.model.module if src.distributed.is_initialized() else self.model),
                optimizer=self.optimizer,
                input_encoder=self.input_encoder,
                epoch=self.epoch,
                step=self.step,
                comet_ml_experiment_key=self.comet_ml.get_key())
            if checkpoint.path.exists():
                return None
            return checkpoint.save()
        else:
            return None

    @log_runtime
    def run_epoch(self, train=False, trial_run=False, infer=False):
        """ Iterate over a dataset with ``self.model``, computing the loss function every iteration.

        TODO: In PyTorch 1.2 they allow DDP gradient accumulation to further increase training
        speed, try it.

        Args:
            train (bool, optional): If ``True`` the model will additionally take steps along the
                computed gradient; furthermore, the Trainer ``step`` and ``epoch`` state will be
                updated.
            trial_run (bool, optional): If ``True`` then the epoch is limited to one batch.
            infer (bool, optional): If ``True`` the model is run in inference mode.
        """
        if infer and train:
            raise ValueError('Train and infer are mutually exclusive.')

        if train:
            label = self.TRAIN_LABEL
        elif not train and infer:
            label = self.DEV_INFERRED_LABEL
        elif not train:
            label = self.DEV_LABEL

        logger.info('[%s] Running Epoch %d, Step %d', label.upper(), self.epoch, self.step)
        if trial_run:
            logger.info('[%s] Trial run with one batch.', label.upper())

        # Set mode(s)
        self.model.train(mode=train)
        self.comet_ml.set_context(label)
        if not trial_run:
            self.comet_ml.log_current_epoch(self.epoch)

        # NOTE: The `dev_loader` does not always load the same batches. That said, the batches
        # are sampled from the same distribution via `self.dev_dataset`; therefore, it should be
        # comparable between experiments.
        loader_kwargs = {'device': self.device, 'input_encoder': self.input_encoder}
        if train and not hasattr(self, '_train_loader'):
            # NOTE: We cache the `DataLoader` between epochs for performance.
            self._train_loader = DataLoader(self.train_dataset, self.train_batch_size,
                                            **loader_kwargs)
        elif not train and not hasattr(self, '_dev_loader'):
            self._dev_loader = DataLoader(self.dev_dataset, self.dev_batch_size, **loader_kwargs)
        data_loader = self._train_loader if train else self._dev_loader

        random_batch = random.randint(0, len(data_loader) - 1)
        for i, batch in enumerate(data_loader):
            with torch.set_grad_enabled(train):
                if infer:
                    # NOTE: Remove predictions that diverged (reached max) as to not skew other
                    # metrics. We count these sequences seperatly with `reached_max_frames`.
                    predictions = self.model(
                        batch.text.tensor,
                        batch.speaker.tensor,
                        batch.text.lengths,
                        filter_reached_max=True)

                    if predictions[-2].numel() > 0:
                        # NOTE: `average_relative_speed` computes the total length of all predicted
                        # audio over the total length of ground truth audio. In effect, it measures
                        # the model's average speed relative to the original lengths.
                        reached_max_filter = ~predictions[-1].squeeze()
                        lengths = batch.spectrogram.lengths[:, reached_max_filter]
                        self.metrics['average_relative_speed'].update(
                            predictions[-2].sum().float() / lengths.sum().float(), lengths.sum())

                        device = predictions[-2].device
                        self._update_loudness_metrics(
                            batch.spectrogram.tensor[:, reached_max_filter], predictions[1],
                            batch.spectrogram_mask.tensor[:, reached_max_filter],
                            lengths_to_mask(predictions[-2], device=device).transpose(0, 1))

                    self.metrics['reached_max_frames'].update(predictions[-1].float().mean(),
                                                              predictions[-1].numel())
                else:
                    predictions = self.model(batch.text.tensor, batch.speaker.tensor,
                                             batch.text.lengths, batch.spectrogram.tensor,
                                             batch.spectrogram.lengths)
                    self._do_loss_and_maybe_backwards(batch, predictions, do_backwards=train)
                predictions = [p.detach() if torch.is_tensor(p) else p for p in predictions]
                spectrogram_lengths = predictions[-2] if infer else batch.spectrogram.lengths
                self._add_attention_metrics(predictions[3], spectrogram_lengths)

            # NOTE: This metric should be a positive integer indicating that the `data_loader`
            # is loading faster than the data is getting ingested; otherwise, the `data_loader`
            # is bottlenecking training by loading too slowly.
            if hasattr(data_loader.iterator, '_data_queue'):
                self.metrics['data_queue_size'].update(data_loader.iterator._data_queue.qsize())

            if not train and not infer and (i == random_batch or trial_run):
                self._visualize_predicted(batch, predictions)

            for name, metric in self.metrics.items():
                self.comet_ml.log_metric('step/%s' % name, metric.sync().last_update())
            self._log_loudness_metrics('step/', lambda m: m.sync().last_update())

            if train:
                self.step += 1
                self.comet_ml.set_step(self.step)
                self.scheduler.step(self.step)

            if trial_run:
                break

        # Log epoch metrics
        if not trial_run:
            self.comet_ml.log_epoch_end(self.epoch)
            for name, metric in self.metrics.items():
                self.comet_ml.log_metric('epoch/%s' % name, metric.sync().reset())
            self._log_loudness_metrics('epoch/', lambda m: m.sync().reset())
            if train:
                self.epoch += 1
        else:
            for _, metric in itertools.chain(self.metrics.items(), self.loudness_metrics.items()):
                metric.reset()

    def _get_loudness(self, spectrogram, mask=None, **kwargs):
        """ Compute the loudness from a spectrogram.

        Args:
            spectrogram (torch.FloatTensor [num_frames, batch_size, frame_channels])
            mask (torch.FloatTensor [num_frames, batch_size])
            **kwargs: Additional key word arguments passed to `framed_rms_from_power_spectrogram`.

        Returns:
            torch.FloatTensor [1]: The loudness in decibels of the spectrogram.
        """
        device = spectrogram.device
        spectrogram = db_to_power(spectrogram.transpose(0, 1))
        target_rms = framed_rms_from_power_spectrogram(spectrogram, **kwargs)
        mask = torch.ones(
            *target_rms.shape, device=device) if mask is None else mask.transpose(0, 1)

        # TODO: This conversion from framed RMS to global RMS is not accurate. The original
        # spectrogram is padded such that it's length is some constant multiple (256x) of the signal
        # length. In order to accurately convert a framed RMS to a global RMS, each sample
        # has to appear an equal number of times in the frames. Supposing there is 25% overlap,
        # that means each sample has to appear 4 times. That nessecarly means that the first and
        # last sample needs to be evaluated 3x times, adding 6 frames to the total number of frames.
        # Adding a constant number of frames, is not compatible with a constant multiple supposing
        # any sequence length must be supported. To fix this, we need to remove the requirement
        # for a constant multiple. The requirement comes from the signal model that upsamples
        # via constant multiples at the moment. We could adjust the signal model so that
        # it upsamples -6 frames then a constant multiple, each time. A change like this
        # would ensure that the first and last frame have 4x overlapping frames to describe
        # the audio sequence, increase performance at the boundary.
        return power_to_db((target_rms * mask).pow(2).sum() / (mask.sum()))

    def _update_loudness_metrics(self,
                                 target,
                                 predicted,
                                 target_mask=None,
                                 predicted_mask=None,
                                 **kwargs):
        """ Update the `self.loudness_metrics`.

        Args:
            target (torch.FloatTensor [num_frames, batch_size, frame_channels]): Target spectrogram.
            predicted (torch.FloatTensor [num_frames, batch_size, frame_channels]): Predicted
                spectrogram.
            target_mask (torch.FloatTensor [num_frames, batch_size]): Target spectrogram mask.
            predicted_mask (torch.FloatTensor [num_frames, batch_size]): Predicted spectrogram mask.
            **kwargs: Additional key word arguments passed to `self._get_loudness`.
        """
        target_rms = db_to_power(self._get_loudness(target, target_mask, **kwargs))
        predicted_rms = db_to_power(self._get_loudness(predicted, predicted_mask, **kwargs))
        self.loudness_metrics['average_target_loudness'].update(
            target_rms,
            target.numel() if target_mask is None else target_mask.sum())
        self.loudness_metrics['average_predicted_loudness'].update(
            predicted_rms,
            predicted_rms.numel() if predicted_mask is None else predicted_mask.sum())

    def _log_loudness_metrics(self, prefix, get_metric):
        """ Log to `comet_ml` `self.loudness_metrics`.

        Args:
            prefix (str)
            get_metric (callable): Callable run on the values of `self.loudness_metrics`.
        """
        target = get_metric(self.loudness_metrics['average_target_loudness'])
        target = power_to_db(torch.tensor(target)) if target is not None else target
        predicted = get_metric(self.loudness_metrics['average_predicted_loudness'])
        predicted = power_to_db(torch.tensor(predicted)) if predicted is not None else predicted
        delta = predicted - target if target is not None and predicted is not None else None
        self.comet_ml.log_metric('%saverage_target_loudness' % prefix, target)
        self.comet_ml.log_metric('%saverage_predicted_loudness' % prefix, predicted)
        self.comet_ml.log_metric('%saverage_delta_loudness' % prefix, delta)

    @configurable
    def _do_loss_and_maybe_backwards(self,
                                     batch,
                                     predictions,
                                     do_backwards,
                                     stop_threshold=HParam()):
        """ Compute the losses and maybe do backwards.

        TODO: Consider logging seperate metrics per speaker.

        Args:
            batch (SpectrogramModelTrainingRow)
            predictions (any): Return value from ``self.model.forwards``.
            do_backwards (bool): If ``True`` backward propogate the loss.
            stop_threshold (float): The threshold probability for deciding to stop.
        """
        # predicted_pre_spectrogram, predicted_post_spectrogram
        # [num_frames, batch_size, frame_channels]
        # predicted_stop_tokens [num_frames, batch_size]
        # predicted_alignments [num_frames, batch_size, num_tokens]
        (predicted_pre_spectrogram, predicted_post_spectrogram, predicted_stop_tokens,
         predicted_alignments) = predictions
        spectrogram = batch.spectrogram.tensor  # [num_frames, batch_size, frame_channels]

        # expanded_mask [num_frames, batch_size, frame_channels]
        expanded_mask = batch.spectrogram_expanded_mask.tensor
        # pre_spectrogram_loss [num_frames, batch_size, frame_channels]
        pre_spectrogram_loss = self.criterion_spectrogram(predicted_pre_spectrogram, spectrogram)
        # [num_frames, batch_size, frame_channels] → [num_frames, batch_size, frame_channels]
        pre_spectrogram_loss = pre_spectrogram_loss * expanded_mask

        # post_spectrogram_loss [num_frames, batch_size, frame_channels]
        post_spectrogram_loss = self.criterion_spectrogram(predicted_post_spectrogram, spectrogram)
        # [num_frames, batch_size, frame_channels] → [num_frames, batch_size, frame_channels]
        post_spectrogram_loss = post_spectrogram_loss * expanded_mask

        mask = batch.spectrogram_mask.tensor  # [num_frames, batch_size]
        # stop_token_loss [num_frames, batch_size]
        stop_token_loss = self.criterion_stop_token(predicted_stop_tokens, batch.stop_token.tensor)
        # [num_frames, batch_size] → [num_frames, batch_size]
        stop_token_loss = stop_token_loss * mask

        if do_backwards:
            self.optimizer.zero_grad()
            # NOTE: We sum over the `num_frames` dimension to ensure that we don't bias based on
            # `num_frames`. For example, a larger `num_frames` means that the average denominator
            # is larger; therefore, the loss value for each element is smaller.
            # NOTE: We average accross `batch_size` and `frame_channels` so that the loss
            # stays around the same value regardless of the `batch_size`. This should not
            # affect convergence because both of these are constant values; however, this should
            # help normalize the loss value between experiments with different `batch_size` and
            # `frame_channels`.

            # pre_spectrogram_loss [num_frames, batch_size, frame_channels] → [1]
            # post_spectrogram_loss [num_frames, batch_size, frame_channels] → [1]
            # stop_token_loss [num_frames, batch_size] → [1]
            expected_average_spectrogram_length = (
                self._train_loader.expected_average_spectrogram_length)
            # NOTE: The loss is calibrated to match the loss of older models. Without this
            # calibration, the model doesn't train well.
            # TODO: Parameterize these loss scalars.
            ((pre_spectrogram_loss.sum(dim=0) / expected_average_spectrogram_length).mean() / 100 +
             (post_spectrogram_loss.sum(dim=0) / expected_average_spectrogram_length).mean() / 100 +
             (stop_token_loss.sum(dim=0) / expected_average_spectrogram_length).mean() /
             4).backward()
            self.optimizer.step(comet_ml=self.comet_ml)

        expected_stop_token = (batch.stop_token.tensor > stop_threshold).masked_select(mask > 0)
        predicted_stop_token = (torch.sigmoid(predicted_stop_tokens) >
                                stop_threshold).masked_select(mask > 0)

        self.metrics['stop_token_accuracy'].update(
            (expected_stop_token == predicted_stop_token).float().mean(), mask.sum())

        # NOTE: These metrics track the average loss per tensor element.
        # NOTE: These losses are from the original Tacotron 2 paper.
        self.metrics['pre_spectrogram_loss'].update(pre_spectrogram_loss.mean(),
                                                    expanded_mask.sum())
        self.metrics['post_spectrogram_loss'].update(post_spectrogram_loss.mean(),
                                                     expanded_mask.sum())
        self.metrics['stop_token_loss'].update(stop_token_loss.mean(), mask.sum())

        return (pre_spectrogram_loss.sum(), post_spectrogram_loss.sum(), stop_token_loss.sum(),
                expanded_mask.sum(), mask.sum())

    def _add_attention_metrics(self, predicted_alignments, lengths):
        """ Compute and report attention metrics.

        Args:
            predicted_alignments (torch.FloatTensor [num_frames, batch_size, num_tokens])
            lengths (torch.LongTensor [batch_size])
        """
        if lengths.numel() == 0:
            return  # No need to update our metrics

        # lengths [batch_size] → mask [batch_size, num_frames]
        mask = lengths_to_mask(lengths, device=predicted_alignments.device)
        # mask [batch_size, num_frames] → [num_frames, batch_size]
        mask = mask.transpose(0, 1)
        kwargs = {'tensor': predicted_alignments, 'dim': 2, 'mask': mask}
        # NOTE: `attention_norm` with `norm=math.inf` computes the maximum value along `num_tokens`
        # dimension.
        self.metrics['attention_norm'].update(
            get_average_norm(norm=math.inf, **kwargs), kwargs['mask'].sum())
        # NOTE: `attention_std` computes the standard deviation along `num_tokens` dimension.
        self.metrics['attention_std'].update(get_weighted_stdev(**kwargs), kwargs['mask'].sum())

    def visualize_inferred(self):
        """ Run in inference mode and visualize results.
        """
        if not src.distributed.is_master():
            return

        example = random.sample(self.dev_dataset, 1)[0]
        text, speaker = tensors_to(
            self.input_encoder.encode((example.text, example.speaker)), device=self.device)
        model = self.model.module if src.distributed.is_initialized() else self.model

        with evaluate(model, device=self.device):
            logger.info('Running inference...')
            (predicted_pre_spectrogram, predicted_post_spectrogram, predicted_stop_tokens,
             predicted_alignments, _, _) = model(text, speaker)

        predicted_residual = predicted_post_spectrogram - predicted_pre_spectrogram
        gold_spectrogram = maybe_load_tensor(example.spectrogram)
        gold_loudness = self._get_loudness(gold_spectrogram).cpu().item()
        predicted_loudness = self._get_loudness(predicted_post_spectrogram).cpu().item()

        self.comet_ml.set_context(self.DEV_INFERRED_LABEL)
        logged_audio = {
            'predicted_griffin_lim_audio': griffin_lim(predicted_post_spectrogram.cpu().numpy()),
            'gold_griffin_lim_audio': griffin_lim(gold_spectrogram.numpy()),
            'gold_audio': maybe_load_tensor(example.spectrogram_audio)
        }
        self.comet_ml.log_audio(
            audio=logged_audio,
            tag=self.DEV_INFERRED_LABEL,
            text=example.text,
            speaker=example.speaker,
            predicted_loudness=predicted_loudness,
            gold_loudness=gold_loudness)
        self.comet_ml.log_metrics({  # [num_frames, num_tokens] → scalar
            'single/attention_norm': get_average_norm(predicted_alignments, dim=1, norm=math.inf),
            'single/attention_std': get_weighted_stdev(predicted_alignments, dim=1),
            'single/target_loudness': gold_loudness,
            'single/predicted_loudness': predicted_loudness,
            'single/delta_loudness': predicted_loudness - gold_loudness,
        })
        self.comet_ml.log_figures({
            'final_spectrogram': plot_mel_spectrogram(predicted_post_spectrogram),
            'residual_spectrogram': plot_mel_spectrogram(predicted_residual),
            'gold_spectrogram': plot_mel_spectrogram(gold_spectrogram),
            'pre_spectrogram': plot_mel_spectrogram(predicted_pre_spectrogram),
            'alignment': plot_attention(predicted_alignments),
            'stop_token': plot_stop_token(predicted_stop_tokens),
        })

    def _visualize_predicted(self, batch, predictions):
        """ Visualize examples from a batch.

        Args:
            batch (SpectrogramModelTrainingRow)
            predictions (any): Return value from ``self.model.forwards``.
        """
        (predicted_pre_spectrogram, predicted_post_spectrogram, predicted_stop_tokens,
         predicted_alignments) = predictions
        batch_size = predicted_post_spectrogram.shape[1]
        item = random.randint(0, batch_size - 1)
        spectrogram_length = int(batch.spectrogram.lengths[0, item].item())
        text_length = int(batch.text.lengths[0, item].item())

        # predicted_post_spectrogram [num_frames, frame_channels]
        predicted_post_spectrogram = predicted_post_spectrogram[:spectrogram_length, item]
        # predicted_pre_spectrogram [num_frames, frame_channels]
        predicted_pre_spectrogram = predicted_pre_spectrogram[:spectrogram_length, item]
        # gold_spectrogram [num_frames, frame_channels]
        gold_spectrogram = batch.spectrogram.tensor[:spectrogram_length, item]

        # [num_frames, frame_channels] → [num_frames]
        post_spectrogram_loss_per_frame = self.criterion_spectrogram(predicted_post_spectrogram,
                                                                     gold_spectrogram).mean(dim=1)
        # [num_frames, frame_channels] → [num_frames]
        pre_spectrogram_loss_per_frame = self.criterion_spectrogram(predicted_pre_spectrogram,
                                                                    gold_spectrogram).mean(dim=1)

        predicted_residual = predicted_post_spectrogram - predicted_pre_spectrogram
        predicted_delta = abs(gold_spectrogram - predicted_post_spectrogram)

        predicted_alignments = predicted_alignments[:spectrogram_length, item, :text_length]
        predicted_stop_tokens = predicted_stop_tokens[:spectrogram_length, item]

        self.comet_ml.log_metrics({  # [num_frames, num_tokens] → scalar
            'single/attention_norm': get_average_norm(predicted_alignments, dim=1, norm=math.inf),
            'single/attention_std': get_weighted_stdev(predicted_alignments, dim=1),
        })
        self.comet_ml.log_figures({
            'post_spectrogram_loss_per_frame': plot_loss_per_frame(post_spectrogram_loss_per_frame),
            'pre_spectrogram_loss_per_frame': plot_loss_per_frame(pre_spectrogram_loss_per_frame),
            'final_spectrogram': plot_mel_spectrogram(predicted_post_spectrogram),
            'residual_spectrogram': plot_mel_spectrogram(predicted_residual),
            'delta_spectrogram': plot_mel_spectrogram(predicted_delta),
            'gold_spectrogram': plot_mel_spectrogram(gold_spectrogram),
            'pre_spectrogram': plot_mel_spectrogram(predicted_pre_spectrogram),
            'alignment': plot_attention(predicted_alignments),
            'stop_token': plot_stop_token(predicted_stop_tokens),
        })

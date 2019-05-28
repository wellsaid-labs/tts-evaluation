import logging
import math
import random

from torchnlp.utils import lengths_to_mask
from torchnlp.utils import tensors_to

import torch

from src.audio import griffin_lim
from src.bin.train.spectrogram_model.data_loader import DataLoader
from src.datasets import add_spectrogram_column
from src.hparams import configurable
from src.hparams import ConfiguredArg
from src.hparams import get_config
from src.optimizers import AutoOptimizer
from src.optimizers import Optimizer
from src.spectrogram_model import InputEncoder
from src.spectrogram_model import SpectrogramModel
from src.utils import AccumulatedMetrics
from src.utils import balance_list
from src.utils import Checkpoint
from src.utils import dict_collapse
from src.utils import evaluate
from src.utils import get_average_norm
from src.utils import get_total_parameters
from src.utils import get_weighted_stdev
from src.utils import log_runtime
from src.visualize import CometML
from src.visualize import plot_attention
from src.visualize import plot_spectrogram
from src.visualize import plot_stop_token

import src.distributed

logger = logging.getLogger(__name__)


class Trainer():
    """ Trainer defines a simple interface for training the ``SpectrogramModel``.

    Args:
        device (torch.device): Device to train on.
        train_dataset (iterable of TextSpeechRow): Train dataset used to optimize the model.
        dev_dataset (iterable of TextSpeechRow): Dev dataset used to evaluate the model.
        comet_ml_project_name (str): Comet project name, used for grouping experiments.
        train_batch_size (int): Batch size used for training.
        dev_batch_size (int): Batch size used for evaluation.
        criterion_spectrogram (callable): Loss function used to score frame predictions.
        criterion_stop_token (callable): Loss function used to score stop
            token predictions.
        optimizer (torch.optim.Optimizer): Optimizer used for gradient descent.
        comet_ml_experiment_key (str, optional): Previous experiment key to continue visualization
            in comet.
        input_encoder (src.spectrogram_model.InputEncoder): Spectrogram model input encoder.
        model (torch.nn.Module, optional): Model to train and evaluate.
        step (int, optional): Starting step; typically, this parameter is useful when starting from
            a checkpoint.
        epoch (int, optional): Starting epoch; typically, this parameter is useful when starting
            from a checkpoint.
        use_tqdm (bool, optional): Use TQDM to track epoch progress.
    """

    TRAIN_LABEL = 'train'
    DEV_INFERRED_LABEL = 'dev_inferred'
    DEV_LABEL = 'dev'

    @configurable
    def __init__(self,
                 device,
                 train_dataset,
                 dev_dataset,
                 comet_ml_project_name,
                 train_batch_size=ConfiguredArg(),
                 dev_batch_size=ConfiguredArg(),
                 criterion_spectrogram=ConfiguredArg(),
                 criterion_stop_token=ConfiguredArg(),
                 optimizer=ConfiguredArg(),
                 comet_ml_experiment_key=None,
                 input_encoder=None,
                 model=None,
                 step=0,
                 epoch=0,
                 use_tqdm=False):

        self.train_dataset = add_spectrogram_column(train_dataset)
        # The training and development dataset distribution of speakers is arbitrary (i.e. some
        # audio books have more data and some have less). In order to ensure that no speaker
        # is prioritized over another, we balance the number of examples for each speaker.
        self.dev_dataset = add_spectrogram_column(
            balance_list(dev_dataset, get_class=lambda r: r.speaker, random_seed=123))

        self.input_encoder = InputEncoder(
            [r.text for r in self.train_dataset],
            [r.speaker for r in self.train_dataset]) if input_encoder is None else input_encoder

        # Allow for ``class`` or a class instance
        self.model = model if isinstance(model, torch.nn.Module) else SpectrogramModel(
            self.input_encoder.text_encoder.vocab_size,
            self.input_encoder.speaker_encoder.vocab_size)
        self.model.to(device)
        if src.distributed.is_initialized():
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model, device_ids=[device], output_device=device, dim=1)

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

        # Comet creates an experiment on `comet.ml`; therefore, this takes some time to execute.
        # We leave this till the very end; ensursing everything else works before
        # creating a comet_ml experiment.
        self.comet_ml = CometML(
            project_name=comet_ml_project_name,
            experiment_key=comet_ml_experiment_key,
            disabled=not src.distributed.is_master())
        self.comet_ml.set_step(step)
        self.comet_ml.log_current_epoch(epoch)
        self.comet_ml.log_dataset_hash([self.train_dataset, self.dev_dataset])
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
        })

        logger.info('Training on %d GPUs', torch.cuda.device_count())
        logger.info('Step: %d', self.step)
        logger.info('Vocab: %s', sorted(self.input_encoder.text_encoder.vocab))
        logger.info('Epoch: %d', self.epoch)
        logger.info('Train Batch Size: %d', train_batch_size)
        logger.info('Dev Batch Size: %d', dev_batch_size)
        logger.info('Model:\n%s', self.model)
        logger.info('Is Comet ML disabled? %s', 'True' if self.comet_ml.disabled else 'False')

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
            'comet_ml_experiment_key': checkpoint.comet_ml_experiment_key,
            'input_encoder': checkpoint.input_encoder,
            'comet_ml_project_name': checkpoint.comet_ml_project_name
        }
        checkpoint_kwargs.update(kwargs)
        return class_(**checkpoint_kwargs)

    def save_checkpoint(self, directory):
        """ Save a checkpoint.

        Args:
            directory (str): Directory to store checkpoint

        Returns:
            (str): Path the checkpoint was saved to.
        """
        return Checkpoint(
            comet_ml_project_name=self.comet_ml.project_name,
            directory=directory,
            model=(self.model.module if src.distributed.is_initialized() else self.model),
            optimizer=self.optimizer,
            input_encoder=self.input_encoder,
            epoch=self.epoch,
            step=self.step,
            comet_ml_experiment_key=self.comet_ml.get_key()).save()

    @log_runtime
    def run_epoch(self, train=False, trial_run=False, infer=False):
        """ Iterate over a dataset with ``self.model``, computing the loss function every iteration.

        Args:
            train (bool, optional): If ``True`` the model will additionally take steps along the
                computed gradient; furthermore, the Trainer ``step`` and ``epoch`` state will be
                updated.
            trial_run (bool, optional): If ``True`` then the epoch is limited to one batch.
            infer (bool): If ``True`` the model is run in inference mode.
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

        # Setup iterator and metrics
        dataset = balance_list(
            self.train_dataset, get_class=lambda r: r.speaker) if train else self.dev_dataset
        data_loader = DataLoader(
            data=dataset,
            batch_size=self.train_batch_size if train else self.dev_batch_size,
            device=self.device,
            input_encoder=self.input_encoder,
            trial_run=trial_run,
            use_tqdm=self.use_tqdm)

        # Run epoch
        # NOTE: Within a distributed execution, ``random.randint`` produces different values in
        # different processes. For example, the master process generator may be ahead of the
        # worker processes because it executes auxiliary code the workers do not.
        random_batch = random.randint(0, len(data_loader) - 1)
        for i, batch in enumerate(data_loader):
            with torch.set_grad_enabled(train):
                if infer:
                    predictions = self.model(batch.text[0], batch.speaker[0], batch.text[1])
                    self.accumulated_metrics.add_metric(
                        'duration_gap',
                        (predictions[-1].float() / batch.spectrogram[1].float()).mean(),
                        predictions[-1].numel())
                else:
                    predictions = self.model(batch.text[0], batch.speaker[0], batch.text[1],
                                             batch.spectrogram[0], batch.spectrogram[1])
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

        self.accumulated_metrics.reset()

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
        self.accumulated_metrics.add_metrics({
            'pre_spectrogram_loss': pre_spectrogram_loss,
            'post_spectrogram_loss': post_spectrogram_loss,
        }, expanded_mask.sum())
        self.accumulated_metrics.add_metrics({'stop_token_loss': stop_token_loss}, mask.sum())

        return (pre_spectrogram_loss, post_spectrogram_loss, stop_token_loss, expanded_mask.sum(),
                mask.sum())

    def _add_attention_metrics(self, predicted_alignments, lengths):
        """ Compute and report attention metrics """
        # predicted_alignments [num_frames, batch_size, num_tokens]
        mask = lengths_to_mask(lengths, device=predicted_alignments.device).transpose(0, 1)
        kwargs = {'tensor': predicted_alignments.detach(), 'dim': 2, 'mask': mask}
        self.accumulated_metrics.add_metrics({
            'attention_norm': get_average_norm(norm=math.inf, **kwargs),
            'attention_std': get_weighted_stdev(**kwargs),
        }, kwargs['mask'].sum())

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
             predicted_alignments, _) = model(text, speaker)

        predicted_residual = predicted_post_spectrogram - predicted_pre_spectrogram

        self.comet_ml.set_context(self.DEV_INFERRED_LABEL)
        self.comet_ml.log_audio(
            tag=self.DEV_INFERRED_LABEL,
            text=example.text,
            speaker=example.speaker,
            predicted_audio=griffin_lim(predicted_post_spectrogram.cpu().numpy()),
            gold_audio=example.spectrogram_audio.to_tensor())
        self.comet_ml.log_metrics({  # [num_frames, num_tokens] → scalar
            'single/attention_norm': get_average_norm(predicted_alignments, dim=1, norm=math.inf),
            'single/attention_std': get_weighted_stdev(predicted_alignments, dim=1),
        })
        self.comet_ml.log_figures({
            'final_spectrogram': plot_spectrogram(predicted_post_spectrogram),
            'residual_spectrogram': plot_spectrogram(predicted_residual),
            'gold_spectrogram': plot_spectrogram(example.spectrogram.to_tensor()),
            'pre_spectrogram': plot_spectrogram(predicted_pre_spectrogram),
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
        spectrogam_length = int(batch.spectrogram[1][0, item].item())
        text_length = int(batch.text[1][0, item].item())

        predicted_post_spectrogram = predicted_post_spectrogram[:spectrogam_length, item]
        predicted_pre_spectrogram = predicted_pre_spectrogram[:spectrogam_length, item]
        gold_spectrogram = batch.spectrogram[0][:spectrogam_length, item]

        predicted_residual = predicted_post_spectrogram - predicted_pre_spectrogram
        predicted_delta = abs(gold_spectrogram - predicted_post_spectrogram)

        predicted_alignments = predicted_alignments[:spectrogam_length, item, :text_length]
        predicted_stop_tokens = predicted_stop_tokens[:spectrogam_length, item]

        self.comet_ml.log_metrics({  # [num_frames, num_tokens] → scalar
            'single/attention_norm': get_average_norm(predicted_alignments, dim=1, norm=math.inf),
            'single/attention_std': get_weighted_stdev(predicted_alignments, dim=1),
        })
        self.comet_ml.log_figures({
            'final_spectrogram': plot_spectrogram(predicted_post_spectrogram),
            'residual_spectrogram': plot_spectrogram(predicted_residual),
            'delta_spectrogram': plot_spectrogram(predicted_delta),
            'gold_spectrogram': plot_spectrogram(gold_spectrogram),
            'pre_spectrogram': plot_spectrogram(predicted_pre_spectrogram),
            'alignment': plot_attention(predicted_alignments),
            'stop_token': plot_stop_token(predicted_stop_tokens),
        })

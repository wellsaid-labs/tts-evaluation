from collections import defaultdict

import atexit
import logging
import math
import random

from torchnlp.utils import lengths_to_mask
from torchnlp.utils import tensors_to

import torch

from src.audio import griffin_lim
from src.bin.train.spectrogram_model.data_loader import DataLoader
from src.hparams import configurable
from src.hparams import ConfiguredArg
from src.hparams import get_config
from src.optimizers import AutoOptimizer
from src.optimizers import Optimizer
from src.spectrogram_model import InputEncoder
from src.utils import balance_list
from src.utils import Checkpoint
from src.utils import dict_collapse
from src.utils import DistributedAveragedMetric
from src.utils import evaluate
from src.utils import get_average_norm
from src.utils import get_total_parameters
from src.utils import get_weighted_stdev
from src.utils import log_runtime
from src.utils import maybe_load_tensor
from src.visualize import CometML
from src.visualize import plot_attention
from src.visualize import plot_spectrogram
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
        train_batch_size (int): Batch size used for training.
        dev_batch_size (int): Batch size used for evaluation.
        criterion_spectrogram (callable): Loss function used to score frame predictions.
        criterion_stop_token (callable): Loss function used to score stop
            token predictions.
        optimizer (torch.optim.Optimizer): Optimizer used for gradient descent.
        model (torch.nn.Module, optional): Model to train and evaluate.
        input_encoder (src.spectrogram_model.InputEncoder): Spectrogram model input encoder.
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
                 checkpoints_directory,
                 train_batch_size=ConfiguredArg(),
                 dev_batch_size=ConfiguredArg(),
                 criterion_spectrogram=ConfiguredArg(),
                 criterion_stop_token=ConfiguredArg(),
                 optimizer=ConfiguredArg(),
                 model=ConfiguredArg(),
                 input_encoder=None,
                 step=0,
                 epoch=0,
                 use_tqdm=False):

        self.train_dataset = train_dataset
        # The training and development dataset distribution of speakers is arbitrary (i.e. some
        # audio books have more data and some have less). In order to ensure that no speaker
        # is prioritized over another, we balance the number of examples for each speaker.
        self.dev_dataset = balance_list(dev_dataset, get_class=lambda r: r.speaker, random_seed=123)

        self.input_encoder = InputEncoder(
            [r.text for r in self.train_dataset] +
            [r.text for r in self.dev_dataset], [r.speaker for r in self.train_dataset] +
            [r.speaker for r in self.dev_dataset]) if input_encoder is None else input_encoder

        # Allow for ``class`` or a class instance
        self.model = model if isinstance(model, torch.nn.Module) else model(
            self.input_encoder.text_encoder.vocab_size,
            self.input_encoder.speaker_encoder.vocab_size)
        self.model.to(device)
        if src.distributed.is_initialized():
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model, device_ids=[device], output_device=device, dim=1)

        self.optimizer = optimizer if isinstance(optimizer, Optimizer) else AutoOptimizer(
            optimizer(params=filter(lambda p: p.requires_grad, self.model.parameters())))
        self.optimizer.to(device)

        self.metrics = defaultdict(DistributedAveragedMetric)

        self.device = device
        self.step = step
        self.epoch = epoch
        self.train_batch_size = train_batch_size
        self.dev_batch_size = dev_batch_size
        self.use_tqdm = use_tqdm
        self.checkpoints_directory = checkpoints_directory

        self.criterion_spectrogram = criterion_spectrogram(reduction='none').to(self.device)
        self.criterion_stop_token = criterion_stop_token(reduction='none').to(self.device)

        self.comet_ml = CometML(disabled=not src.distributed.is_master())
        self.comet_ml.set_step(step)
        self.comet_ml.log_current_epoch(epoch)
        self.comet_ml.log_dataset_hash([self.train_dataset, self.dev_dataset])
        self.comet_ml.log_parameters(dict_collapse(get_config()))
        self.comet_ml.set_model_graph(str(self.model))
        train_text_length = sum([len(r.text) for r in self.train_dataset])
        train_text_length = None if len(
            self.train_dataset) == 0 else train_text_length / len(self.train_dataset)
        self.comet_ml.log_parameters({
            'num_parameter': get_total_parameters(self.model),
            'num_training_row': len(self.train_dataset),
            'num_dev_row': len(self.dev_dataset),
            'vocab_size': self.input_encoder.text_encoder.vocab_size,
            'vocab': sorted(self.input_encoder.text_encoder.vocab),
            'num_speakers': self.input_encoder.speaker_encoder.vocab_size,
            'speakers': sorted([str(v) for v in self.input_encoder.speaker_encoder.vocab]),
            'train_text_length': train_text_length
        })
        self._comet_ml_log_input_dev_data_hash()

        logger.info('Training on %d GPUs', torch.cuda.device_count())
        logger.info('Step: %d', self.step)
        logger.info('Vocab: %s', sorted(self.input_encoder.text_encoder.vocab))
        logger.info('Epoch: %d', self.epoch)
        logger.info('Train Batch Size: %d', train_batch_size)
        logger.info('Dev Batch Size: %d', dev_batch_size)
        logger.info('Model:\n%s', self.model)
        logger.info('Is Comet ML disabled? %s', 'True' if self.comet_ml.disabled else 'False')

        atexit.register(self.save_checkpoint)

    def _comet_ml_log_input_dev_data_hash(self, max_examples=10):
        """ Log to comet a basic hash of the predicted spectrogram data in `self.dev_dataset`.

        The predicted spectrogram data varies with the random state and checkpoint; therefore, the
        hash helps differentiate between different datasets.

        Args:
            max_examples (int): The max number of examples to consider for computing the hash. This
                is limited to a small number of elements for faster performance.
        """
        # NOTE: On different GPUs this may produce different results. For example, a V100 computed
        # this value as -63890.96875 while a P100 computed -63890.9625.
        sample = self.dev_dataset[:min(len(self.dev_dataset), max_examples)]
        sum_ = sum([maybe_load_tensor(e.spectrogram).sum() for e in sample])
        average = sum_.item() / len(sample) if len(sample) > 0 else 0.0
        self.comet_ml.log_other('input_dev_data_hash', average)

        # NOTE: Unlike the above hash, this hash should stay stable but will have less variance.
        text_sum = sum([len(e.text) for e in sample])
        text_average = text_sum / len(sample) if len(sample) > 0 else 0.0
        self.comet_ml.log_other('input_dev_text_data_hash', text_average)

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
            (str): Path the checkpoint was saved to.
        """
        return Checkpoint(
            comet_ml_project_name=self.comet_ml.project_name,
            directory=self.checkpoints_directory,
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
                    duration_gap = (predictions[-1].float() / batch.spectrogram[1].float()).mean()
                    self.metrics['duration_gap'].update(duration_gap, predictions[-1].numel())
                else:
                    predictions = self.model(batch.text[0], batch.speaker[0], batch.text[1],
                                             batch.spectrogram[0], batch.spectrogram[1])
                    self._do_loss_and_maybe_backwards(batch, predictions, do_backwards=train)
                predictions = [p.detach() if torch.is_tensor(p) else p for p in predictions]
                spectrogram_lengths = predictions[-1] if infer else batch.spectrogram[1]
                self._add_attention_metrics(predictions[3], spectrogram_lengths)

            if not train and not infer and i == random_batch:
                self._visualize_predicted(batch, predictions)

            for name, metric in self.metrics.items():
                self.comet_ml.log_metric('step/%s' % name, metric.sync().last_update())

            if train:
                self.step += 1
                self.comet_ml.set_step(self.step)

        # Log epoch metrics
        if not trial_run:
            self.comet_ml.log_epoch_end(self.epoch)
            for name, metric in self.metrics.items():
                self.comet_ml.log_metric('epoch/%s' % name, metric.sync().reset())
            if train:
                self.epoch += 1

    def _do_loss_and_maybe_backwards(self, batch, predictions, do_backwards):
        """ Compute the losses and maybe do backwards.

        TODO: Consider logging seperate metrics per speaker.

        Args:
            batch (SpectrogramModelTrainingRow)
            predictions (any): Return value from ``self.model.forwards``.
            do_backwards (bool): If ``True`` backward propogate the loss.
        """
        # predicted_pre_spectrogram, predicted_post_spectrogram
        # [num_frames, batch_size, frame_channels]
        # predicted_stop_tokens [num_frames, batch_size]
        # predicted_alignments [num_frames, batch_size, num_tokens]
        (predicted_pre_spectrogram, predicted_post_spectrogram, predicted_stop_tokens,
         predicted_alignments) = predictions
        spectrogram = batch.spectrogram[0]  # [num_frames, batch_size, frame_channels]

        # expanded_mask [num_frames, batch_size, frame_channels]
        expanded_mask = batch.spectrogram_expanded_mask[0]
        # pre_spectrogram_loss [num_frames, batch_size, frame_channels]
        pre_spectrogram_loss = self.criterion_spectrogram(predicted_pre_spectrogram, spectrogram)
        # [num_frames, batch_size, frame_channels] → [1]
        pre_spectrogram_loss = pre_spectrogram_loss.masked_select(expanded_mask).mean()

        # post_spectrogram_loss [num_frames, batch_size, frame_channels]
        post_spectrogram_loss = self.criterion_spectrogram(predicted_post_spectrogram, spectrogram)
        # [num_frames, batch_size, frame_channels] → [1]
        post_spectrogram_loss = post_spectrogram_loss.masked_select(expanded_mask).mean()

        mask = batch.spectrogram_mask[0]  # [num_frames, batch_size]
        # stop_token_loss [num_frames, batch_size]
        stop_token_loss = self.criterion_stop_token(predicted_stop_tokens, batch.stop_token[0])
        # [num_frames, batch_size] → [1]
        stop_token_loss = stop_token_loss.masked_select(mask).mean()

        if do_backwards:
            self.optimizer.zero_grad()
            (pre_spectrogram_loss + post_spectrogram_loss + stop_token_loss).backward()
            self.optimizer.step(comet_ml=self.comet_ml)

        self.metrics['pre_spectrogram_loss'].update(pre_spectrogram_loss, expanded_mask.sum())
        self.metrics['post_spectrogram_loss'].update(post_spectrogram_loss, expanded_mask.sum())
        self.metrics['stop_token_loss'].update(stop_token_loss, mask.sum())

        return (pre_spectrogram_loss, post_spectrogram_loss, stop_token_loss, expanded_mask.sum(),
                mask.sum())

    def _add_attention_metrics(self, predicted_alignments, lengths):
        """ Compute and report attention metrics.

        Args:
            predicted_alignments (torch.FloatTensor [num_frames, batch_size, num_tokens])
            lengths (torch.LongTensor [batch_size])
        """
        # lengths [batch_size] → mask [batch_size, num_frames]
        mask = lengths_to_mask(lengths, device=predicted_alignments.device)
        # mask [batch_size, num_frames] → [num_frames, batch_size]
        mask = mask.transpose(0, 1)
        kwargs = {'tensor': predicted_alignments, 'dim': 2, 'mask': mask}
        self.metrics['attention_norm'].update(
            get_average_norm(norm=math.inf, **kwargs), kwargs['mask'].sum())
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
             predicted_alignments, _) = model(text, speaker)

        predicted_residual = predicted_post_spectrogram - predicted_pre_spectrogram

        self.comet_ml.set_context(self.DEV_INFERRED_LABEL)
        self.comet_ml.log_audio(
            tag=self.DEV_INFERRED_LABEL,
            text=example.text,
            speaker=example.speaker,
            predicted_audio=griffin_lim(predicted_post_spectrogram.cpu().numpy()),
            gold_audio=maybe_load_tensor(example.spectrogram_audio))
        self.comet_ml.log_metrics({  # [num_frames, num_tokens] → scalar
            'single/attention_norm': get_average_norm(predicted_alignments, dim=1, norm=math.inf),
            'single/attention_std': get_weighted_stdev(predicted_alignments, dim=1),
        })
        self.comet_ml.log_figures({
            'final_spectrogram': plot_spectrogram(predicted_post_spectrogram),
            'residual_spectrogram': plot_spectrogram(predicted_residual),
            'gold_spectrogram': plot_spectrogram(maybe_load_tensor(example.spectrogram)),
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

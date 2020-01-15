"""
NOTE: Epochs this trainer uses are not formal epochs. For example, in the Linda Johnson dataset the
average clip size is 6.57 seconds while the average example seen by the model is 900 samples
or 0.375 seconds. This epoch means that we've taken a random 900 sample clip from the
13,100 clips in the Linda Johnson dataset.

Walking through the math, a real epoch for the Linda Joshon dataset would be about:
    Number of samples: 2066808000 = (23h * 60 * 60 + 55m * 60 + 17s) * 24000
    This epoch sample size: 11790000 = 13,100 * 900
    Formal epoch is 175x larger: 175 ~ 2066808000 / 11790000
    Number of batches in formal epoch: 35,882 ~ 2066808000 / 64 / 900

Find stats on the Linda Johnson dataset here: https://keithito.com/LJ-Speech-Dataset/
"""
from collections import deque
from collections import namedtuple
from copy import deepcopy

import atexit
import logging
import random

from hparams import configurable
from hparams import get_config
from hparams import HParam
from torch.optim.lr_scheduler import LambdaLR
from torchnlp.random import fork_rng
from torchnlp.utils import get_total_parameters

import torch

from src.audio import combine_signal
from src.audio import split_signal
from src.bin.train.signal_model.data_loader import DataLoader
from src.optimizers import AutoOptimizer
from src.optimizers import Optimizer
from src.utils import AnomalyDetector
from src.utils import Checkpoint
from src.utils import dict_collapse
from src.utils import DistributedAveragedMetric
from src.utils import log_runtime
from src.utils import maybe_load_tensor
from src.utils import mean
from src.utils import random_sample
from src.utils import RepeatTimer
from src.visualize import plot_spectrogram

import src.distributed

logger = logging.getLogger(__name__)

_RollbackTrainerState = namedtuple('_RollbackTrainerState', [
    'step', 'epoch', 'optimizer_state_dict', 'model_state_dict', 'scheduler_state_dict',
    'anomaly_detector'
])


class Trainer():
    """ Trainer defines a simple interface for training the ``SignalModel``.

    Args:
        device (torch.device): Device to train on.
        train_dataset (iterable of TextSpeechRow): Train dataset used to optimize the model.
        dev_dataset (iterable of TextSpeechRow): Dev dataset used to evaluate the model.
        checkpoints_directory (str or Path): Directory to store checkpoints in.
        comet_ml (Experiment or ExistingExperiment): Object for visualization with comet.
        train_batch_size (int): Batch size used for training.
        dev_batch_size (int): Batch size used for evaluation.
        criterion (callable): Loss function used to score signal predictions.
        optimizer (torch.optim.Optimizer): Optimizer used for gradient descent.
        min_rollback (int): Minimum number of epochs to rollback in case of a loss anomaly.
        lr_multiplier_schedule (callable): Learning rate multiplier schedule.
        model (torch.nn.Module, optional): Model to train and evaluate.
        spectrogram_model_checkpoint_path (pathlib.Path or str, optional): Checkpoint path used to
            generate a spectrogram from text as input to the signal model.
        step (int, optional): Starting step; typically, this parameter is useful when starting from
            a checkpoint.
        epoch (int, optional): Starting epoch; typically, this parameter is useful when starting
            from a checkpoint.
        num_rollbacks (int, optional): Number of rollbacks.
        anomaly_detector (AnomalyDetector, optional): Anomaly detector used to skip batches that
            result in an anomalous loss.
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
                 train_spectrogram_slice_size=HParam(),
                 dev_batch_size=HParam(),
                 dev_spectrogram_slice_size=HParam(),
                 criterion=HParam(),
                 optimizer=HParam(),
                 min_rollback=HParam(),
                 lr_multiplier_schedule=HParam(),
                 model=HParam(),
                 spectrogram_model_checkpoint_path=None,
                 step=0,
                 epoch=0,
                 num_rollbacks=0,
                 anomaly_detector=None,
                 save_temp_checkpoint_every_n_seconds=60 * 10,
                 dataset_sample_size=50):
        self.device = device
        self.step = step
        self.epoch = epoch
        self.train_batch_size = train_batch_size
        self.train_spectrogram_slice_size = train_spectrogram_slice_size
        self.dev_batch_size = dev_batch_size
        self.dev_spectrogram_slice_size = dev_spectrogram_slice_size
        self.num_rollbacks = num_rollbacks
        self.checkpoints_directory = checkpoints_directory
        self.use_predicted = spectrogram_model_checkpoint_path is not None
        self.spectrogram_model_checkpoint_path = spectrogram_model_checkpoint_path
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset

        self.model = model if isinstance(model, torch.nn.Module) else model()
        self.model.to(device)
        if src.distributed.is_initialized():
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model, device_ids=[device], output_device=device, dim=1)

        self.optimizer = optimizer if isinstance(optimizer, Optimizer) else AutoOptimizer(
            optimizer(params=filter(lambda p: p.requires_grad, self.model.parameters())))
        self.optimizer.to(device)

        self.scheduler = LambdaLR(
            self.optimizer.optimizer, lr_multiplier_schedule, last_epoch=step - 1)

        self.anomaly_detector = anomaly_detector if isinstance(
            anomaly_detector, AnomalyDetector) else AnomalyDetector()

        self.criterion = criterion(reduction='none').to(device)

        self.metrics = {
            'coarse_loss': DistributedAveragedMetric(),
            'fine_loss': DistributedAveragedMetric(),
            'orthogonal_loss': DistributedAveragedMetric(),
            'data_queue_size': DistributedAveragedMetric(),
        }

        # NOTE: Rollback `maxlen=min_rollback + 1` to store the current state of the model with
        # the additional rollbacks.
        self._rollback_states = deque([self._make_partial_rollback_state()],
                                      maxlen=min_rollback + 1)

        self.comet_ml = comet_ml
        self.comet_ml.set_step(step)
        self.comet_ml.log_current_epoch(epoch)
        self.comet_ml.log_parameters(dict_collapse(get_config()))
        self.comet_ml.set_model_graph(str(self.model))
        self.comet_ml.log_parameters({
            'num_parameter': get_total_parameters(self.model),
            'num_training_row': len(self.train_dataset),
            'num_dev_row': len(self.dev_dataset),
        })
        self.comet_ml.log_parameters({
            'expected_average_train_spectrogram_sum':
                self._get_expected_average_spectrogram_sum(self.train_dataset, dataset_sample_size),
            'expected_average_dev_spectrogram_sum':
                self._get_expected_average_spectrogram_sum(self.dev_dataset, dataset_sample_size)
        })
        self.comet_ml.log_other('spectrogram_model_checkpoint_path',
                                str(self.spectrogram_model_checkpoint_path))

        logger.info('Training on %d GPUs', torch.cuda.device_count())
        logger.info('Step: %d', self.step)
        logger.info('Epoch: %d', self.epoch)
        logger.info('Train Batch Size: %d', train_batch_size)
        logger.info('Dev Batch Size: %d', dev_batch_size)
        logger.info('Model:\n%s' % self.model)
        logger.info('Is Comet ML disabled? %s', 'True' if self.comet_ml.disabled else 'False')

        if src.distributed.is_master():
            self.timer = RepeatTimer(save_temp_checkpoint_every_n_seconds,
                                     self._save_checkpoint_repeat_timer)
            self.timer.daemon = True
            self.timer.start()
            atexit.register(self.save_checkpoint)

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
                use_predicted = self.use_predicted
                sample = [
                    r.predicted_spectrogram if use_predicted else r.spectrogram for r in sample
                ]
                return mean(maybe_load_tensor(r).sum().item() for r in sample)
        return None

    def _save_checkpoint_repeat_timer(self):
        """ Save a checkpoint and delete the last checkpoint saved.
        """
        # TODO: Consider using the GCP shutdown scripts via
        # https://haggainuchi.com/shutdown.html
        # NOTE: GCP shutdowns do not trigger `atexit`; therefore, it's useful to always save
        # a temporary checkpoint just in case.
        checkpoint_path = self.save_checkpoint()
        if (hasattr(self, '_last_repeat_timer_checkpoint') and
                self._last_repeat_timer_checkpoint is not None and
                self._last_repeat_timer_checkpoint.exists() and checkpoint_path is not None):
            logger.info('Unlinking temporary checkpoint: %s',
                        str(self._last_repeat_timer_checkpoint))
            self._last_repeat_timer_checkpoint.unlink()
        self._last_repeat_timer_checkpoint = checkpoint_path

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
            # NOTE: ``rollback`` is not checkpointed due to it's size.
            'model': checkpoint.model,
            'optimizer': checkpoint.optimizer,
            'epoch': checkpoint.epoch,
            'step': checkpoint.step,
            'anomaly_detector': checkpoint.anomaly_detector,
            'spectrogram_model_checkpoint_path': checkpoint.spectrogram_model_checkpoint_path,
            'num_rollbacks': checkpoint.num_rollbacks
        }
        checkpoint_kwargs.update(kwargs)
        return class_(**checkpoint_kwargs)

    @log_runtime
    def save_checkpoint(self):
        """ Save a checkpoint.

        Returns:
            (pathlib.Path or None): Path the checkpoint was saved to or None if checkpoint wasn't
                saved.
        """
        if src.distributed.is_master():
            checkpoint = Checkpoint(
                directory=self.checkpoints_directory,
                step=self.step,
                model=(self.model.module if src.distributed.is_initialized() else self.model),
                optimizer=self.optimizer,
                epoch=self.epoch,
                anomaly_detector=self.anomaly_detector,
                comet_ml_project_name=self.comet_ml.project_name,
                comet_ml_experiment_key=self.comet_ml.get_key(),
                spectrogram_model_checkpoint_path=self.spectrogram_model_checkpoint_path,
                num_rollbacks=self.num_rollbacks)
            if checkpoint.path.exists():
                return None
            return checkpoint.save()
        else:
            return None

    def _make_partial_rollback_state(self):
        """ Make a state for rolling back to.

        The rollback includes:
           * Model weights
           * Optimizer weights
           * Step counter `self.step`
           * Epoch counter `self.epoch`

        The rollback does not include:
           * Number of rollbacks `self.num_rollbacks`.
           * Comet_ml `self.comet_ml`. Comet ML does not have a mechanism for rolling back state
             https://github.com/comet-ml/issue-tracking/issues/137.
           * Metrics `self.metrics`. Any metrics are reset during a rollback and a rollback is
             treated as the start of a new epoch.
           * Random Generators. We do not make an effort to perserve the global state of the random
             generators in `torch`, `numpy` and `random` modules.

        Returns:
            (_RollbackTrainerState)
        """
        # TODO: Test to ensure PyTorch operations do not change the state as a result of tensor
        # side effects.
        # NOTE: In PyTorch, unless we ``deepcopy``, ``state_dict`` continues to update.
        return _RollbackTrainerState(
            step=self.step,
            epoch=self.epoch,
            optimizer_state_dict=deepcopy(self.optimizer.state_dict()),
            model_state_dict=deepcopy(self.model.state_dict()),
            scheduler_state_dict=deepcopy(self.scheduler.state_dict()),
            anomaly_detector=deepcopy(self.anomaly_detector))

    def _partial_rollback(self):
        """ Rollback to the earliest state available `self.rollback[0]` and restart the epoch.
        """
        self._end_epoch()

        state = self._rollback_states[0]
        logger.info('Rolling back from step %d to %d and from epoch %d to %d', self.step,
                    state.step, self.epoch, state.epoch)
        self.num_rollbacks += 1
        self.comet_ml.log_metric('num_rollback', self.num_rollbacks)

        self.model.load_state_dict(state.model_state_dict)
        self.optimizer.load_state_dict(state.optimizer_state_dict)
        self.scheduler.load_state_dict(state.scheduler_state_dict)
        self.optimizer.zero_grad()
        self.anomaly_detector = state.anomaly_detector
        self.step = state.step
        self.epoch = state.epoch
        self.comet_ml.set_step(self.step)

        self._rollback_states.clear()  # Clear the future states
        self._rollback_states.append(state)

    def _end_epoch(self):
        """ Reset the trainer state from the current epoch.
        """
        self.comet_ml.log_epoch_end(self.epoch)
        for name, metric in self.metrics.items():
            self.comet_ml.log_metric('epoch/%s' % name, metric.sync().reset())

    @log_runtime
    def run_epoch(self, train=False, trial_run=False):
        """ Iterate over a dataset with ``self.model``, computing the loss function every iteration.

        The specification of an "epoch" is loose in rare circumstances:

            - The trainer allows for partial rollbacks of state during training. There doesn't
              exist effective mechanisms to capture the state of all modules and rollback like the
              `DataLoader` and `comet_ml`.
            - `DataLoader`'s specification allows it to drop data via `drop_last`. Therefore,
              there is not always the same number of batches for each epoch.
            - `trial_run` runs only on row of data.

        Args:
            train (bool, optional): If ``True`` the model will additionally take steps along the
                computed gradient; furthermore, the Trainer ``step`` and ``epoch`` state will be
                updated.
            trial_run (bool, optional): If ``True`` then the epoch is limited to one batch.
        """
        label = self.TRAIN_LABEL if train else self.DEV_LABEL
        if trial_run:
            logger.info('[%s] Trial run with one batch.', label.upper())
        else:
            logger.info('[%s] Running Epoch %d, Step %d', label.upper(), self.epoch, self.step)

        # Set mode(s)
        self.model.train(mode=train)
        self.comet_ml.set_context(label)
        if not trial_run:
            self.comet_ml.log_current_epoch(self.epoch)

        loader_kwargs = {'device': self.device, 'use_predicted': self.use_predicted}
        if train and not hasattr(self, '_train_loader'):
            # NOTE: We cache the `DataLoader` between epochs for performance.
            self._train_loader = DataLoader(
                self.train_dataset,
                self.train_batch_size,
                spectrogram_slice_size=self.train_spectrogram_slice_size,
                **loader_kwargs)
        elif not train and not hasattr(self, '_dev_loader'):
            self._dev_loader = DataLoader(
                self.dev_dataset,
                self.dev_batch_size,
                spectrogram_slice_size=self.dev_spectrogram_slice_size,
                **loader_kwargs)
        data_loader = self._train_loader if train else self._dev_loader

        for i, batch in enumerate(data_loader):
            with torch.set_grad_enabled(train):
                predictions = self.model(
                    batch.input_spectrogram,
                    input_signal=batch.input_signal,
                    target_coarse=batch.target_signal_coarse.unsqueeze(2))
                self._do_loss_and_maybe_backwards(batch, predictions, do_backwards=train)
                predictions = [p.detach() if torch.is_tensor(p) else p for p in predictions]

            # NOTE: This metric should be a positive integer indicating that the `data_loader`
            # is loading faster than the data is getting ingested; otherwise, the `data_loader`
            # is bottlenecking training by loading too slowly.
            if hasattr(data_loader.iterator, 'data_queue'):
                self.metrics['data_queue_size'].update(data_loader.iterator.data_queue.qsize())

            for name, metric in self.metrics.items():
                self.comet_ml.log_metric('step/%s' % name, metric.sync().last_update())

            if train:
                self.step += 1
                self.comet_ml.set_step(self.step)
                self.scheduler.step(self.step)

            if trial_run:
                break

        if not trial_run:
            self._end_epoch()
            if train:
                self.epoch += 1
        else:
            for _, metric in self.metrics.items():
                metric.reset()

    def _get_gru_orthogonal_loss(self):
        """ Get the orthogonal loss for the hidden-to-hidden matrix in our GRUs.

        Papers describing the loss:
        https://papers.nips.cc/paper/7680-can-we-gain-more-from-orthogonality-regularizations-in-training-deep-networks.pdf
        https://github.com/pytorch/pytorch/issues/2421#issuecomment-355534285
        http://mathworld.wolfram.com/FrobeniusNorm.html
        https://github.com/MingtaoGuo/BigGAN-tensorflow/blob/7e531cd875236544866f54248aa397f9176296b6/ops.py#L111

        Returns:
            (torch.FloatTensor [1])
        """
        total = torch.tensor(0.0).to(self.device)
        for name, parameter in self.model.named_parameters():
            if 'gru.weight_hh' in name:
                splits = parameter.chunk(3)
                for split in splits:
                    eye = torch.eye(split.shape[0], device=self.device)
                    total += torch.nn.functional.mse_loss(torch.mm(split, torch.t(split)), eye)
                    total += torch.nn.functional.mse_loss(torch.mm(torch.t(split), split), eye)
        return total

    def _do_loss_and_maybe_backwards(self, batch, predictions, do_backwards):
        """ Compute the losses and maybe do backwards.

        Args:
            batch (SignalModelTrainingRow)
            predictions (any): Return value from ``self.model.forwards``.
            do_backwards (bool): If ``True`` backward propogate the loss.
        """
        (predicted_coarse, predicted_fine, _) = predictions

        # [batch_size, signal_length, bins] → [batch_size, bins, signal_length]
        predicted_fine = predicted_fine.transpose(1, 2)
        predicted_coarse = predicted_coarse.transpose(1, 2)

        # coarse_loss [batch_size, signal_length]
        coarse_loss = self.criterion(predicted_coarse, batch.target_signal_coarse)
        coarse_loss = coarse_loss.masked_select(batch.signal_mask).mean()  # coarse_loss [1]

        # fine_loss [batch_size, signal_length]
        fine_loss = self.criterion(predicted_fine, batch.target_signal_fine)
        fine_loss = fine_loss.masked_select(batch.signal_mask).mean()  # fine_loss [1]

        orthogonal_loss = self._get_gru_orthogonal_loss()  # orthogonal_loss [1]

        if do_backwards:
            self.optimizer.zero_grad()
            (coarse_loss + fine_loss).backward()
            grad_norm = self.optimizer.step(comet_ml=self.comet_ml)

            grad_norm = torch.tensor(grad_norm, device=coarse_loss.device)
            torch.distributed.all_reduce(grad_norm)
            grad_norm = (grad_norm / torch.distributed.get_world_size()).cpu().item()
            if self.anomaly_detector.step(grad_norm):
                logger.warning('Rolling back, detected an anomaly #%d (%f > %f ± %f)',
                               self.num_rollbacks, grad_norm, self.anomaly_detector.last_average,
                               self.anomaly_detector.max_deviation)
                self._partial_rollback()
            else:
                self._rollback_states.append(self._make_partial_rollback_state())

        # Learn more about `coarse_loss` and `fine_loss` in the "Efficient Neural Audio Synthesis"
        # paper.
        self.metrics['coarse_loss'].update(coarse_loss, batch.signal_mask.sum())
        self.metrics['fine_loss'].update(fine_loss, batch.signal_mask.sum())
        self.metrics['orthogonal_loss'].update(orthogonal_loss)

        return coarse_loss, fine_loss, batch.signal_mask.sum()

    def _get_sample_density_gap(self, predicted_signal, target_signal, greater_than):
        """ Measure the relative difference in sample density inside a specified range.

        Args:
            predicted_signal (torch.IntTensor [predicted_signal_length])
            target_signal (torch.IntTensor [target_signal_length])
            greater_than (float): Defines the amplitude range.

        Returns:
            (float): The density margin between the predicted and target signals.
        """

        def get_density(unnormalized_signal):
            maximum = torch.iinfo(unnormalized_signal.dtype).max
            normalized = unnormalized_signal.float() / maximum
            num_selected_samples = (normalized.abs() >= greater_than).sum()
            total_samples = unnormalized_signal.numel()
            return num_selected_samples.float() / total_samples

        return (get_density(predicted_signal) - get_density(target_signal)).item()

    @log_runtime
    def visualize_inferred(self):
        """ Run in inference mode and visualize results.
        """
        if not src.distributed.is_master():
            return

        self.comet_ml.set_context(self.DEV_INFERRED_LABEL)
        model = self.model.module if src.distributed.is_initialized() else self.model

        example = random.sample(self.dev_dataset, 1)[0]
        spectrogram = example.predicted_spectrogram if self.use_predicted else example.spectrogram
        spectrogram = maybe_load_tensor(spectrogram)  # [num_frames, frame_channels]
        target_signal = maybe_load_tensor(example.spectrogram_audio).float()  # [signal_length]
        spectrogram = spectrogram.to(torch.device('cpu'))

        logger.info('Running inference on %d spectrogram frames with %d threads.',
                    spectrogram.shape[0], torch.get_num_threads())

        inferrer = model.to_inferrer()
        predicted_coarse, predicted_fine, _ = inferrer(spectrogram)
        predicted = combine_signal(predicted_coarse, predicted_fine, return_int=True)

        # NOTE: Introduce quantization noise similar to the model outputs and inputs.
        target = combine_signal(*split_signal(target_signal), return_int=True)

        self.comet_ml.log_metrics({
            'single/%s_sample_density_gap' % str(int(n * 100)):
            self._get_sample_density_gap(predicted, target, n) for n in [0.99, 0.95, 0.90]
        })
        self.comet_ml.log_audio(
            tag=self.DEV_INFERRED_LABEL,
            text=example.text,
            speaker=str(example.speaker),
            gold_audio=target,
            predicted_audio=predicted)
        self.comet_ml.log_figure('spectrogram', plot_spectrogram(spectrogram))

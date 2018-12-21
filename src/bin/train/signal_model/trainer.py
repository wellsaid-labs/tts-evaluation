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
from functools import partial

import logging
import random
import time

from torch.nn import CrossEntropyLoss
from torch.optim import Adam

import torch

from src.audio import combine_signal
from src.audio import split_signal
from src.bin.train.signal_model.data_loader import DataLoader
from src.datasets import compute_spectrograms
from src.hparams import configurable
from src.hparams import get_config
from src.optimizer import AutoOptimizer
from src.optimizer import Optimizer
from src.signal_model import WaveRNN
from src.utils import AnomalyDetector
from src.utils import dict_collapse
from src.utils import evaluate
from src.utils import get_total_parameters
from src.utils import OnDiskTensor
from src.visualize import AccumulatedMetrics
from src.visualize import CometML
from src.visualize import plot_spectrogram

logger = logging.getLogger(__name__)

# TODO: Remove default parameters that are configured; otherwise, it's confusing as to which
# values are used. Make a default ``configured_parameter`` object to set as a default for configured
# parameters. This object throws an error if it's used in any shape or form.
# Also, the configurable, can check for that parameter. Or the parameter can check to ensure it's
# actually configured.

TrainerState = namedtuple('TrainerState',
                          ['step', 'epoch', 'optimizer_state_dict', 'model_state_dict'])


class Trainer():
    """ Trainer that manages model training (i.e. running epochs, logging, etc.).

    Args:
        device (torch.device): Device to train on.
        train_dataset (iterable): Train dataset used to optimize the model.
        dev_dataset (iterable): Dev dataset used to evaluate.
        comet_ml_project_name (str): Project name, used for grouping experiments.
        comet_ml_experiment_key (str, optioanl): Previous experiment key to restart visualization.
        spectrogram_model_checkpoint_path (str, optional): Checkpoint used to generate a spectrogram
            from text as input to the signal model.
        train_batch_size (int, optional): Batch size used for training.
        dev_batch_size (int, optional): Batch size used for evaluation.
        model (torch.nn.Module, optional): Model to train and evaluate.
        step (int, optional): Starting step, useful warm starts (i.e. checkpoints).
        epoch (int, optional): Starting epoch, useful warm starts (i.e. checkpoints).
        step_unit (str, optional): Unit to measuer steps in, either: ['batches', 'deciseconds'].
        criterion (callable): Loss function used to score signal predictions.
        optimizer (torch.optim.Optimizer): Optimizer used for gradient descent.
        anomaly_detector (AnomalyDetector, optional): Anomaly detector used to skip batches that
            result in large loss.
        min_rollback (int, optional): Minimum number of epochs to rollback in case of an anomaly.
        use_tqdm (bool, optional): Use TQDM to track epoch progress.
        use_predicted (bool, optional): If ``True`` train from predicted spectrogram, otherwise
            train from real spectrogram.
    """

    STEP_UNIT_DECISECONDS = 'deciseconds'
    STEP_UNIT_BATCHES = 'batches'

    @configurable
    def __init__(self,
                 device,
                 train_dataset,
                 dev_dataset,
                 comet_ml_project_name,
                 comet_ml_experiment_key=None,
                 spectrogram_model_checkpoint_path=None,
                 train_batch_size=32,
                 dev_batch_size=128,
                 model=WaveRNN,
                 step=0,
                 epoch=0,
                 step_unit='batches',
                 criterion=CrossEntropyLoss,
                 optimizer=Adam,
                 anomaly_detector=None,
                 min_rollback=1,
                 use_tqdm=False,
                 use_predicted=True):
        assert step_unit in [self.STEP_UNIT_BATCHES,
                             self.STEP_UNIT_DECISECONDS], 'Picked invalid step unit.'

        # Allow for ``class`` or a class instance
        self.model = model if isinstance(model, torch.nn.Module) else model()
        self.model.to(device)

        self.optimizer = optimizer if isinstance(optimizer, Optimizer) else AutoOptimizer(
            optimizer(params=filter(lambda p: p.requires_grad, self.model.parameters())))
        self.optimizer.to(device)

        self.anomaly_detector = anomaly_detector if isinstance(
            anomaly_detector, AnomalyDetector) else AnomalyDetector()

        self.criterion = criterion(reduction='none').to(device)
        self.accumulated_metrics = AccumulatedMetrics()

        self.device = device
        self.step = step
        self.epoch = epoch
        self.step_unit = step_unit
        self.train_batch_size = train_batch_size
        self.dev_batch_size = dev_batch_size
        self.spectrogram_model_checkpoint_path = spectrogram_model_checkpoint_path
        compute_spectrograms_partial = partial(
            compute_spectrograms,
            checkpoint_path=spectrogram_model_checkpoint_path,
            device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        self.train_dataset = compute_spectrograms_partial(train_dataset)
        self.dev_dataset = compute_spectrograms_partial(dev_dataset)
        self.use_tqdm = use_tqdm
        self.use_predicted = use_predicted
        self.random = random.Random(123)  # Ensure the same samples are sampled
        # NOTE: Rollback ``maxlen=min_rollback + 1`` to store the current state of the model with
        # the additional rollbacks.
        self.rollback = deque([self._get_state()], maxlen=min_rollback + 1)

        self.comet_ml = CometML(
            project_name=comet_ml_project_name, experiment_key=comet_ml_experiment_key)
        self.comet_ml.set_step(step)
        self.comet_ml.log_current_epoch(epoch)
        self.comet_ml.log_dataset_hash([self.train_dataset, self.dev_dataset])
        self.comet_ml.log_parameters(dict_collapse(get_config()))
        self.comet_ml.set_model_graph(str(self.model))
        self.comet_ml.log_parameters({
            'num_parameter': get_total_parameters(self.model),
            'num_training_row': len(self.train_dataset),
            'num_dev_row': len(self.dev_dataset),
        })
        self.comet_ml.log_other('spectrogram_model_checkpoint_path',
                                spectrogram_model_checkpoint_path)

        logger.info('Training on %d GPUs', torch.cuda.device_count())
        logger.info('Step (%s): %d', self.step_unit, self.step)
        logger.info('Epoch: %d', self.epoch)
        logger.info('Train Batch Size: %d', train_batch_size)
        logger.info('Dev Batch Size: %d', dev_batch_size)
        logger.info('Model:\n%s' % self.model)

    def _get_state(self):
        # TODO: Add a test to ensure that _get_state does not change after run_epoch or something
        # NOTE: In PyTorch, unless we ``deepcopy``, ``state_dict`` continues to update.
        return TrainerState(
            step=self.step,
            epoch=self.epoch,
            optimizer_state_dict=deepcopy(self.optimizer.state_dict()),
            model_state_dict=deepcopy(self.model.state_dict()))

    def _set_state(self, state):
        self.model.load_state_dict(state.model_state_dict)
        self.optimizer.load_state_dict(state.optimizer_state_dict)
        self.step = state.step
        self.epoch = state.epoch

    def _maybe_rollback(self, epoch_coarse_loss):
        """ Maybe rollback the model if the loss is too high.

        Args:
            epoch_coarse_loss (float)
        """
        is_anomaly = self.anomaly_detector.step(epoch_coarse_loss)
        if is_anomaly:
            logger.warning('Rolling back, detected a coarse loss anomaly #%d (%f > %f ± %f)',
                           self.anomaly_detector.anomaly_counter, epoch_coarse_loss,
                           self.anomaly_detector.last_average, self.anomaly_detector.max_deviation)
            state = self.rollback[0]
            logger.info('Rolling back from step %d to %d and from epoch %d to %d', self.step,
                        state.step, self.epoch, state.epoch)
            self._set_state(state)

            # Clear the possibly degenerative states.
            self.rollback.clear()
            self.rollback.append(state)
        else:
            self.rollback.append(self._get_state())

    def run_epoch(self, train=False, trial_run=False):
        """ Iterate over a dataset with ``self.model``, computing the loss function every iteration.

        Args:
            train (bool, optional): If ``True``, the batch will store gradients.
            trial_run (bool, optional): If ``True``, then runs only 1 batch.
        """
        label = 'TRAIN' if train else 'DEV'
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
            trial_run=trial_run,
            random=self.random,
            use_predicted=self.use_predicted,
            use_tqdm=self.use_tqdm)

        # Run epoch
        start_time = time.time()
        for i, batch in enumerate(data_loader):
            with torch.set_grad_enabled(train):
                predictions = torch.nn.parallel.data_parallel(
                    module=self.model,
                    inputs=batch.input_spectrogram[0],
                    module_kwargs={
                        'input_signal': batch.input_signal[0],
                        'target_coarse': batch.target_signal_coarse[0].unsqueeze(2)
                    })
                self._do_loss_and_maybe_backwards(batch, predictions, do_backwards=train)
                predictions = [p.detach() if torch.is_tensor(p) else p for p in predictions]

            self.accumulated_metrics.log_step_end(
                lambda k, v: self.comet_ml.log_metric('step/' + k, v) if train else None)

            if train:
                if self.step_unit == self.STEP_UNIT_DECISECONDS:
                    self.step += time.time() - start_time
                    self.comet_ml.set_step(int(round(self.step * 10)))
                elif self.step_unit == self.STEP_UNIT_BATCHES:
                    self.step += 1
                    self.comet_ml.set_step(self.step)

        # Log epoch metrics
        if not trial_run:
            self.comet_ml.log_epoch_end(self.epoch)
            if train:
                self.epoch += 1
                self._maybe_rollback(self.accumulated_metrics.get_epoch_metric('coarse_loss'))
            self.accumulated_metrics.log_epoch_end(
                lambda k, v: self.comet_ml.log_metric('epoch/' + k, v))

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
        coarse_loss = self.criterion(predicted_coarse, batch.target_signal_coarse[0])
        coarse_loss = coarse_loss.masked_select(batch.signal_mask[0]).mean()

        # fine_loss [batch_size, signal_length]
        fine_loss = self.criterion(predicted_fine, batch.target_signal_fine[0])
        fine_loss = fine_loss.masked_select(batch.signal_mask[0]).mean()

        if do_backwards:
            self.optimizer.zero_grad()
            (coarse_loss + fine_loss).backward()
            self.optimizer.step(comet_ml=self.comet_ml)

        # Record metrics
        self.accumulated_metrics.add_multiple_metrics({
            'coarse_loss': coarse_loss,
            'fine_loss': fine_loss
        }, batch.signal_mask[0].sum())

        return coarse_loss, fine_loss, batch.signal_mask[0].sum()

    def visualize_infered(self):
        """ Run in inference mode without teacher forcing and push results to Tensorboard.

        Returns: None
        """
        example = random.sample(self.dev_dataset, 1)[0]
        spectrogram = example.predicted_spectrogram if self.use_predicted else example.spectrogram
        # [num_frames, frame_channels]
        spectrogram = spectrogram.to_tensor() if isinstance(spectrogram,
                                                            OnDiskTensor) else spectrogram
        # [signal_length]
        target_signal = example.spectrogram_audio.to_tensor() if isinstance(
            example.spectrogram_audio, OnDiskTensor) else example.spectrogram_audio
        # Introduce quantization noise
        target_signal = combine_signal(*split_signal(target_signal))

        inferrer = self.model.to_inferrer(self.device)
        with evaluate(inferrer):
            logger.info('Running inference on %d spectrogram frames...', spectrogram.shape[0])
            predicted_coarse, predicted_fine, _ = inferrer(spectrogram)
            predicted_signal = combine_signal(predicted_coarse, predicted_fine)

        self.comet_ml.log_audio(
            tag='infered',
            text=example.text,
            speaker=str(example.speaker),
            gold_audio=target_signal,
            predicted_audio=predicted_signal)
        self.comet_ml.log_figure('infered/spectrogram', plot_spectrogram(spectrogram))

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
import atexit
import logging
import random
import warnings

from hparams import configurable
from hparams import get_config
from hparams import HParam
from torch.optim.lr_scheduler import LambdaLR
from torchnlp.random import fork_rng
from torchnlp.utils import get_total_parameters

import torch

from src.audio import integer_to_floating_point_pcm
from src.audio import SignalToLogMelSpectrogram
from src.bin.train.signal_model.data_loader import DataLoader
from src.optimizers import AutoOptimizer
from src.optimizers import Optimizer
from src.utils import Checkpoint
from src.utils import dict_collapse
from src.utils import DistributedAveragedMetric
from src.utils import log_runtime
from src.utils import maybe_load_tensor
from src.utils import mean
from src.utils import random_sample
from src.utils import RepeatTimer
from src.visualize import plot_mel_spectrogram
from src.visualize import plot_spectrogram

import src.distributed

logger = logging.getLogger(__name__)


class ExponentialMovingParameterAverage():
    """ Average the model parameters over time.

    Inspired by: http://www.programmersought.com/article/28492072406/

    TODO: Consider moving this into the signal model as part of it's evaluation mode.

    Learn more about EMA, here: https://arxiv.org/abs/1806.04498

    Args:
        model (torch.nn.Module): The model w/ parameters to average.
        beta (float): Beta used to weight the exponential mean.
    """

    def __init__(self, parameters, beta=0.9999):
        self.parameters = list(parameters)
        self.beta = beta
        self.shadow = [param.clone().detach() * (1.0 - self.beta) for param in self.parameters]
        self.backup = []
        self.step = 1

    def update(self):
        """ Update the parameter average.
        """
        for i, param in enumerate(self.parameters):
            self.shadow[i] = (1.0 - self.beta) * param.clone().detach() + self.beta * self.shadow[i]
        self.step += 1

    def apply_shadow(self):
        """ Replace the model with it's averaged parameters.
        """
        self.backup = [param.clone().detach() for param in self.parameters]
        for param, shadow in zip(self.parameters, self.shadow):
            # The initial 0.0 average values introduce bias that is corrected, learn more:
            # https://www.coursera.org/lecture/deep-neural-network/bias-correction-in-exponentially-weighted-averages-XjuhD
            with torch.no_grad():
                param.copy_(shadow / (1 - self.beta**(self.step)))

    def restore(self):
        """ Restore the model's old parameters.
        """
        for param, backup in zip(self.parameters, self.backup):
            with torch.no_grad():
                param.copy_(backup)
        self.backup = []

    def to(self, device):
        """ Move the state to ``device``.

        Args:
            device (torch.device)
        """
        for list_ in [self.parameters, self.shadow, self.backup]:
            for i, param in enumerate(list_):
                list_[i] = param.to(device)
        return self


class SpectrogramLoss(torch.nn.Module):
    """ Compute a loss based in the time / frequency domain.

    TODO: Incorperate `HParams` for parameterizing the loss.

    Args:
        criterion (torch.nn.Module): The loss function for comparing two spectrograms.
        **kwargs: Additional key word arguments passed to `SignalToLogMelSpectrogram`.
    """

    def __init__(self, criterion=torch.nn.L1Loss, **kwargs):
        super().__init__()

        self.signal_to_spectrogram = SignalToLogMelSpectrogram(**kwargs)
        self.fft_length = self.signal_to_spectrogram.fft_length
        self.frame_hop = self.signal_to_spectrogram.frame_hop
        self.sample_rate = self.signal_to_spectrogram.sample_rate

        self.criterion = criterion(reduction='none')
        self.plot_spectrogram = lambda spectrogram: plot_spectrogram(
            spectrogram, frame_hop=self.frame_hop, sample_rate=self.sample_rate)
        self.plot_mel_spectrogram = lambda spectrogram: plot_mel_spectrogram(
            spectrogram, frame_hop=self.frame_hop, sample_rate=self.sample_rate)

    def get_name(self, signal_name=None, is_mel_scale=True, is_decibels=True, is_magnitude=True):
        """
        Args:
            signal_name (str): The same of the signal.
            is_mel_scale (str): If `True` the signal spectrogram was fit to the mel scale.
            is_decibels (str): If `True` the signal spectrogram is on the decibel scale.
            is_magnitude (str): If `True` the signal spectrogram is a magnitude spectrogram.

        Returns:
            (str): A string representing the data type.
        """
        name = '' if signal_name is None else '%s,' % signal_name
        name = 'spectrogram(%sfft_length=%d,frame_hop=%d)' % (name, self.fft_length, self.frame_hop)
        name = 'abs(%s)' % name if is_magnitude else name
        name = 'db(%s)' % name if is_decibels else name
        name = 'mel(%s)' % name if is_mel_scale else name
        return name

    def forward(self, predicted_signal, target_signal, comet_ml=None):
        """
        Args:
            predicted_signal (torch.FloatTensor [batch_size (optional), signal_length])
            target_signal (torch.FloatTensor [batch_size (optional), signal_length])
            comet_ml (None or Experiment): If this value is passed, then this logs figures to
                comet. If the batch size is larger than one, then a random item from the
                batch is picked to be logged.

        Returns:
            torch.FloatTensor: The loss.
        """
        predicted_signal = predicted_signal.view(-1, predicted_signal.shape[-1])
        target_signal = target_signal.view(-1, target_signal.shape[-1])

        (predicted_db_mel_spectrogam, predicted_db_spectrogram,
         predicted_spectrogram) = self.signal_to_spectrogram(
             predicted_signal, intermediate=True)
        (target_db_mel_spectrogam, target_db_spectrogram,
         target_spectrogram) = self.signal_to_spectrogram(
             target_signal, intermediate=True)
        db_mel_spectrogam_loss = self.criterion(predicted_db_mel_spectrogam,
                                                target_db_mel_spectrogam)

        if comet_ml:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    'ignore', module=r'.*hparams', message=r'.*Overwriting configured argument.*')
                batch_size = predicted_signal.shape[0]
                random_item = random.randint(0, batch_size - 1)
                comet_ml.log_figure(
                    self.get_name('predicted'),
                    self.plot_mel_spectrogram(predicted_db_mel_spectrogam[random_item]))
                comet_ml.log_figure(
                    self.get_name('target'),
                    self.plot_mel_spectrogram(target_db_mel_spectrogam[random_item]))
                comet_ml.log_figure(
                    self.get_name('predicted', is_mel_scale=False),
                    self.plot_spectrogram(predicted_db_spectrogram[random_item]))
                comet_ml.log_figure(
                    self.get_name('target', is_mel_scale=False),
                    self.plot_spectrogram(target_db_spectrogram[random_item]))
                comet_ml.log_figure(
                    self.get_name('predicted', is_mel_scale=False, is_decibels=False),
                    self.plot_spectrogram(predicted_spectrogram[random_item]))
                comet_ml.log_figure(
                    self.get_name('target', is_mel_scale=False, is_decibels=False),
                    self.plot_spectrogram(target_spectrogram[random_item]))
                comet_ml.log_figure('%s(%s)' % (self.criterion.__class__.__name__, self.get_name()),
                                    self.plot_mel_spectrogram(db_mel_spectrogam_loss[random_item]))
                comet_ml.log_figure(
                    'mean(%s(%s))' % (self.criterion.__class__.__name__, self.get_name()),
                    self.plot_mel_spectrogram(db_mel_spectrogam_loss[random_item].mean(
                        dim=0, keepdim=True)))

        return db_mel_spectrogam_loss.mean()


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
        lr_multiplier_schedule (callable): Learning rate multiplier schedule.
        model (torch.nn.Module, optional): Model to train and evaluate.
        spectrogram_model_checkpoint_path (pathlib.Path or str, optional): Checkpoint path used to
            generate a spectrogram from text as input to the signal model.
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
                 train_spectrogram_slice_size=HParam(),
                 dev_batch_size=HParam(),
                 dev_spectrogram_slice_size=HParam(),
                 optimizer=HParam(),
                 lr_multiplier_schedule=HParam(),
                 exponential_moving_parameter_average=ExponentialMovingParameterAverage,
                 model=HParam(),
                 spectrogram_model_checkpoint_path=None,
                 step=0,
                 epoch=0,
                 save_temp_checkpoint_every_n_seconds=60 * 10,
                 dataset_sample_size=50):
        self.device = device
        self.step = step
        self.epoch = epoch
        self.train_batch_size = train_batch_size
        self.train_spectrogram_slice_size = train_spectrogram_slice_size
        self.dev_batch_size = dev_batch_size
        self.dev_spectrogram_slice_size = dev_spectrogram_slice_size
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

        self.exponential_moving_parameter_average = (
            exponential_moving_parameter_average
            if isinstance(exponential_moving_parameter_average, ExponentialMovingParameterAverage)
            else exponential_moving_parameter_average(
                filter(lambda p: p.requires_grad, self.model.parameters())))
        self.exponential_moving_parameter_average.to(device)

        self.optimizer = optimizer if isinstance(optimizer, Optimizer) else AutoOptimizer(
            optimizer(params=filter(lambda p: p.requires_grad, self.model.parameters())))
        self.optimizer.to(device)

        self.scheduler = LambdaLR(
            self.optimizer.optimizer, lr_multiplier_schedule, last_epoch=step - 1)

        self.criterion = SpectrogramLoss().to(device)

        # NOTE: The `num_mel_bins` must be proportional to `fft_length`, learn more:
        # https://stackoverflow.com/questions/56929874/what-is-the-warning-empty-filters-detected-in-mel-frequency-basis-about
        with warnings.catch_warnings():
            warnings.filterwarnings(
                'ignore', module=r'.*hparams', message=r'.*Overwriting configured argument.*')
            self.criterions = [
                SpectrogramLoss(
                    fft_length=2048,
                    frame_hop=300,
                    window=torch.hann_window(1200),
                    num_mel_bins=128).to(device),
                SpectrogramLoss(
                    fft_length=1024, frame_hop=150, window=torch.hann_window(600),
                    num_mel_bins=64).to(device),
                SpectrogramLoss(
                    fft_length=512, frame_hop=75, window=torch.hann_window(300),
                    num_mel_bins=32).to(device),
            ]

        # TODO: Remove redundancy between `self.criterions` and `self.metrics`.
        # TODO: Consider naming the metrics more precisely from the spectrogram parameters.
        self.metrics = {
            'log_mel_spectrogram_magnitude_loss': DistributedAveragedMetric(),
            'log_mel_2048_spectrogram_magnitude_loss': DistributedAveragedMetric(),
            'log_mel_1024_spectrogram_magnitude_loss': DistributedAveragedMetric(),
            'log_mel_512_spectrogram_magnitude_loss': DistributedAveragedMetric(),
            'data_queue_size': DistributedAveragedMetric(),
        }

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
            'model': checkpoint.model,
            'optimizer': checkpoint.optimizer,
            'epoch': checkpoint.epoch,
            'step': checkpoint.step,
            'spectrogram_model_checkpoint_path': checkpoint.spectrogram_model_checkpoint_path,
            'exponential_moving_parameter_average': checkpoint.exponential_moving_parameter_average,
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
                comet_ml_project_name=self.comet_ml.project_name,
                comet_ml_experiment_key=self.comet_ml.get_key(),
                spectrogram_model_checkpoint_path=self.spectrogram_model_checkpoint_path,
                exponential_moving_parameter_average=self.exponential_moving_parameter_average)
            if checkpoint.path.exists():
                return None
            return checkpoint.save()
        else:
            return None

    @log_runtime
    def run_epoch(self, train=False, trial_run=False):
        """ Iterate over a dataset with ``self.model``, computing the loss function every iteration.

        The specification of an "epoch" is loose in rare circumstances:

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

        if not train:
            self.exponential_moving_parameter_average.apply_shadow()

        for i, batch in enumerate(data_loader):
            with torch.set_grad_enabled(train):
                predicted_signal = self.model(batch.spectrogram, pad_input=False)
                self._do_loss_and_maybe_backwards(batch, predicted_signal, do_backwards=train)

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

        if not train:
            self.exponential_moving_parameter_average.restore()

        if not trial_run:
            self.comet_ml.log_epoch_end(self.epoch)
            for name, metric in self.metrics.items():
                self.comet_ml.log_metric('epoch/%s' % name, metric.sync().reset())
            if train:
                self.epoch += 1
        else:
            for _, metric in self.metrics.items():
                metric.reset()

    def _do_loss_and_maybe_backwards(self, batch, predicted_signal, do_backwards, log_figure=False):
        """ Compute the losses and maybe do backwards.

        Args:
            batch (SignalModelTrainingRow)
            predicted_signal (torch.FloatTensor [batch_size, signal_length])
            do_backwards (bool): If ``True`` backward propogate the loss.
            log_figure (bool): If `True` this logs figures.
        """
        assert batch.target_signal.shape == predicted_signal.shape, (
            'The shapes do not match %s =!= %s' %
            (batch.target_signal.shape, predicted_signal.shape))

        total_spectrogram_loss = torch.tensor(0.0, device=predicted_signal.device)
        for criterion in self.criterions:
            spectrogram_loss = criterion(
                predicted_signal,
                batch.target_signal,
                comet_ml=self.comet_ml if log_figure else None)
            total_spectrogram_loss += spectrogram_loss / len(self.criterions)
            self.metrics['log_mel_%d_spectrogram_magnitude_loss' % criterion.fft_length].update(
                spectrogram_loss, batch.target_signal.shape[0])

        if do_backwards:
            self.optimizer.zero_grad()
            total_spectrogram_loss.backward()
            self.optimizer.step(comet_ml=self.comet_ml)
            self.exponential_moving_parameter_average.update()

        # TODO: Consider using the spectrogram length instead of batch size
        self.metrics['log_mel_spectrogram_magnitude_loss'].update(total_spectrogram_loss,
                                                                  batch.target_signal.shape[0])

        return total_spectrogram_loss, batch.signal_mask.sum()

    @log_runtime
    def visualize_inferred(self):
        """ Run in inference mode and visualize results.
        """
        if not src.distributed.is_master():
            return

        # TODO: Consider running the algorithm end-to-end with the spectrogram model on CPU to
        # have a end-to-end comparison.
        # TODO: Consider transfer learning the signal model from ground truth to a particular
        # spectrogram model.
        self.comet_ml.set_context(self.DEV_INFERRED_LABEL)
        model = self.model.module if src.distributed.is_initialized() else self.model
        model = model.eval()
        example = random.sample(self.dev_dataset, 1)[0]
        spectrogram = example.predicted_spectrogram if self.use_predicted else example.spectrogram
        spectrogram = maybe_load_tensor(spectrogram)  # [num_frames, frame_channels]
        target_signal = integer_to_floating_point_pcm(maybe_load_tensor(
            example.spectrogram_audio)).to(self.device)  # [signal_length]
        spectrogram = spectrogram.to(self.device)

        logger.info('Running inference on %d spectrogram frames with %d threads.',
                    spectrogram.shape[0], torch.get_num_threads())

        self.exponential_moving_parameter_average.apply_shadow()
        predicted = model(spectrogram)
        self.exponential_moving_parameter_average.restore()

        total_spectrogram_loss = torch.tensor(0.0, device=self.device)
        for criterion in self.criterions:
            spectrogram_loss = criterion(predicted, target_signal, comet_ml=self.comet_ml)
            total_spectrogram_loss += spectrogram_loss / len(self.criterions)
            self.comet_ml.log_metrics({
                'single/log_mel_%d_spectrogram_magnitude_loss' % criterion.fft_length:
                    spectrogram_loss.item()
            })

        self.comet_ml.log_metrics(
            {'single/log_mel_spectrogram_magnitude_loss': total_spectrogram_loss.item()})
        self.comet_ml.log_audio(
            tag=self.DEV_INFERRED_LABEL,
            text=example.text,
            speaker=str(example.speaker),
            gold_audio=target_signal,
            predicted_audio=predicted,
            log_mel_spectrogram_magnitude_loss=total_spectrogram_loss.item())
        self.comet_ml.log_figure('input_spectrogram', plot_spectrogram(spectrogram.detach().cpu()))

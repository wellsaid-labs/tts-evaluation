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
from src.audio import SignalTodBMelSpectrogram
from src.bin.train.signal_model.data_loader import DataLoader
from src.optimizers import AutoOptimizer
from src.optimizers import ExponentialMovingParameterAverage
from src.optimizers import Optimizer
from src.signal_model import generate_waveform
from src.utils import Checkpoint
from src.utils import dict_collapse
from src.utils import DistributedAveragedMetric
from src.utils import evaluate
from src.utils import log_runtime
from src.utils import maybe_load_tensor
from src.utils import mean
from src.utils import random_sample
from src.utils import RepeatTimer
from src.visualize import plot_mel_spectrogram
from src.visualize import plot_spectrogram

import src.distributed

logger = logging.getLogger(__name__)


class SpectrogramLoss(torch.nn.Module):
    """ Compute a loss based in the time / frequency domain.

    NOTE: This loss undersamples boundary samples; therefore, it's important in the training
    data that each sample has an equal chance of being a boundary / none-boundary sample so
    that all samples are undersampled equally.
    NOTE: The loss boundary effect affects samller spectrograms disproportionately.
    NOTE: The loss is average accross the spectrogram frames; therefore, there is no bias for
    the length of the spectrogram.

    Args:
        device (torch.device, optional)
        criterion (torch.nn.Module): The loss function for comparing two spectrograms.
        discriminator (torch.nn.Module): The model used to discriminate between two spectrograms.
        discriminator_optimizer (torch.nn.Module)
        discriminator_criterion (torch.nn.Module)
        **kwargs: Additional key word arguments passed to `SignalTodBMelSpectrogram`.
    """

    @configurable
    def __init__(self,
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                 criterion=HParam(),
                 discriminator=HParam(),
                 discriminator_optimizer=HParam(),
                 discriminator_criterion=HParam(),
                 **kwargs):
        super().__init__()

        # NOTE: The `SpectrogramLoss` has it's own configuration for `SignalTodBMelSpectrogram`.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                'ignore', module=r'.*hparams', message=r'.*Overwriting configured argument.*')
            self.signal_to_spectrogram = SignalTodBMelSpectrogram(**kwargs).to(device)

        self.fft_length = self.signal_to_spectrogram.fft_length
        self.frame_hop = self.signal_to_spectrogram.frame_hop
        self.sample_rate = self.signal_to_spectrogram.sample_rate
        self.num_mel_bins = self.signal_to_spectrogram.num_mel_bins

        self.criterion = criterion(reduction='none').to(device)

        self.discriminator = discriminator(self.fft_length, self.num_mel_bins).to(device)
        if src.distributed.is_initialized():
            self.discriminator = torch.nn.parallel.DistributedDataParallel(
                self.discriminator, device_ids=[device], output_device=device, dim=1)

        discriminator_optimizer = discriminator_optimizer(
            params=filter(lambda p: p.requires_grad, self.discriminator.parameters()))
        self.discriminator_optimizer = Optimizer(discriminator_optimizer).to(device)

        self.discriminator_criterion = discriminator_criterion().to(device)

    def plot_spectrogram(self, *args, **kwargs):
        return plot_spectrogram(
            *args, **kwargs, frame_hop=self.frame_hop, sample_rate=self.sample_rate)

    def plot_mel_spectrogram(self, *args, **kwargs):
        return plot_mel_spectrogram(
            *args, **kwargs, frame_hop=self.frame_hop, sample_rate=self.sample_rate)

    def get_name(self, signal_name=None, is_mel_scale=True, is_decibels=True, is_magnitude=True):
        """ Get a interpretable label for logging a spectrogram.

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

    def discriminate(self,
                     predicted_spectrogram,
                     predicted_db_spectrogram,
                     predicted_db_mel_spectrogram,
                     target_spectrogram,
                     target_db_spectrogram,
                     target_db_mel_spectrogram,
                     do_backwards=False):
        """ Discriminate between predicted and real spectrograms.

        Learn more about this approach:
        https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html

        Args:
            predicted_spectrogram (torch.FloatTensor [batch_size, num_frames, fft_length // 2 + 1])
            predicted_db_spectrogram (torch.FloatTensor
                [batch_size, num_frames, fft_length // 2 + 1])
            predicted_db_mel_spectrogram (torch.FloatTensor [batch_size, num_frames, num_mel_bins])
            target_spectrogram (torch.FloatTensor [batch_size, num_frames, fft_length // 2 + 1])
            target_db_spectrogram (torch.FloatTensor [batch_size, num_frames, fft_length // 2 + 1])
            target_db_mel_spectrogram (torch.FloatTensor [batch_size, num_frames, num_mel_bins])
            do_backwards (bool, optional): If `True` this updates the discriminator weights.

        Returns:
            (torch.FloatTensor): The generator loss on `predicted`.
            (torch.FloatTensor): The discriminator loss on `predicted` and `target`.
            (torch.FloatTensor): The accuracy of the discriminator on `predicted` and `target`.
        """
        batch_size = predicted_spectrogram.shape[0]
        device = predicted_spectrogram.device

        labels_target = torch.full((batch_size,), 1.0, device=device)
        labels_predicted = torch.full((batch_size,), 0.0, device=device)
        labels = torch.cat([labels_target, labels_predicted])

        # NOTE: `detach` to avoid updating the generator.
        db_mel_spectrogram = [target_db_mel_spectrogram, predicted_db_mel_spectrogram.detach()]
        db_mel_spectrogram = torch.cat(db_mel_spectrogram)
        db_spectrogram = torch.cat([target_db_spectrogram, predicted_db_spectrogram.detach()])
        spectrogram = torch.cat([target_spectrogram, predicted_spectrogram.detach()])

        predictions = self.discriminator(spectrogram, db_spectrogram, db_mel_spectrogram)
        loss = self.discriminator_criterion(predictions, labels)
        accuracy = ((labels > 0.5) == (torch.sigmoid(predictions) > 0.5)).float().mean()

        if do_backwards:
            self.discriminator.zero_grad()
            loss.backward()
            self.discriminator_optimizer.step()

        # NOTE: Use target labels instead of predicted to flip the gradient for the generator.
        predictions = self.discriminator(predicted_spectrogram, predicted_db_spectrogram,
                                         predicted_db_mel_spectrogram)
        return self.discriminator_criterion(predictions, labels_target), loss, accuracy

    def forward(self, predicted_signal, target_signal, comet_ml=None, do_backwards=False):
        """
        Args:
            predicted_signal (torch.FloatTensor [batch_size (optional), signal_length])
            target_signal (torch.FloatTensor [batch_size (optional), signal_length])
            comet_ml (None or Experiment): If this value is passed, then this logs figures to
                comet. If the batch size is larger than one, then a random item from the
                batch is picked to be logged.
            do_backwards (bool, optional): If `True` this updates the discriminator weights.

        Returns:
            torch.FloatTensor: The spectrogram loss.
            torch.FloatTensor: The generator loss.
            torch.FloatTensor: The discriminator loss.
            torch.FloatTensor: The discriminator accuracy.
            int: The number of frames.
        """
        predicted_signal = predicted_signal.view(-1, predicted_signal.shape[-1])
        target_signal = target_signal.view(-1, target_signal.shape[-1])

        assert target_signal.shape == predicted_signal.shape

        (predicted_db_mel_spectrogram, predicted_db_spectrogram,
         predicted_spectrogram) = self.signal_to_spectrogram(
             predicted_signal, intermediate=True)
        (target_db_mel_spectrogram, target_db_spectrogram,
         target_spectrogram) = self.signal_to_spectrogram(
             target_signal, intermediate=True)

        db_mel_spectrogram_loss = self.criterion(predicted_db_mel_spectrogram,
                                                 target_db_mel_spectrogram)

        generator_loss, disciminator_loss, disciminator_accuracy = self.discriminate(
            predicted_spectrogram,
            predicted_db_spectrogram,
            predicted_db_mel_spectrogram,
            target_spectrogram,
            target_db_spectrogram,
            target_db_mel_spectrogram,
            do_backwards=do_backwards)

        if comet_ml:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    'ignore', module=r'.*hparams', message=r'.*Overwriting configured argument.*')
                batch_size = predicted_signal.shape[0]
                random_item = random.randint(0, batch_size - 1) if batch_size > 1 else 0
                comet_ml.log_figure(
                    self.get_name('predicted'),
                    self.plot_mel_spectrogram(predicted_db_mel_spectrogram[random_item]))
                comet_ml.log_figure(
                    self.get_name('target'),
                    self.plot_mel_spectrogram(target_db_mel_spectrogram[random_item]))
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
                                    self.plot_mel_spectrogram(db_mel_spectrogram_loss[random_item]))
                comet_ml.log_figure(
                    'mean(%s(%s))' % (self.criterion.__class__.__name__, self.get_name()),
                    self.plot_mel_spectrogram(db_mel_spectrogram_loss[random_item].mean(
                        dim=0, keepdim=True)))

        return (db_mel_spectrogram_loss.mean(), generator_loss, disciminator_loss,
                disciminator_accuracy)


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
        criterions (list of callables, optional): List of callables to initialize criterions.
        criterions_state_dict (list of dict, optional): An optional state dict for each criterion.
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
                 criterions=HParam(),
                 criterions_state_dict=None,
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

        self.criterions = [c(device) for c in criterions]
        if criterions_state_dict is not None:
            list(c.load_state_dict(s) for c, s in zip(self.criterions, criterions_state_dict))

        self.metrics = {
            'data_queue_size': DistributedAveragedMetric(),
            'db_mel_spectrogram_magnitude_loss': DistributedAveragedMetric(),
            'spectrogram_generator_loss': DistributedAveragedMetric(),
            'spectrogram_discriminator_loss': DistributedAveragedMetric(),
            'spectrogram_discriminator_accuracy': DistributedAveragedMetric(),
        }
        self.metrics.update({
            '%d_spectrogram_generator_loss' % c.fft_length: DistributedAveragedMetric()
            for c in self.criterions
        })
        self.metrics.update({
            '%d_spectrogram_discriminator_loss' % c.fft_length: DistributedAveragedMetric()
            for c in self.criterions
        })
        self.metrics.update({
            '%d_spectrogram_discriminator_accuracy' % c.fft_length: DistributedAveragedMetric()
            for c in self.criterions
        })
        self.metrics.update({
            'db_mel_%d_spectrogram_magnitude_loss' % c.fft_length: DistributedAveragedMetric()
            for c in self.criterions
        })

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
            'criterions_state_dict': checkpoint.criterions_state_dict,
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
                exponential_moving_parameter_average=self.exponential_moving_parameter_average,
                criterions_state_dict=[c.state_dict() for c in self.criterions])
            if checkpoint.path.exists():
                return None
            return checkpoint.save()
        else:
            return None

    @log_runtime
    def run_epoch(self, train=False, trial_run=False):
        """ Iterate over a dataset with `self.model`, computing the loss function every iteration.

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
                spectrogram_slice_pad=self.model.padding,
                **loader_kwargs)
        elif not train and not hasattr(self, '_dev_loader'):
            self._dev_loader = DataLoader(
                self.dev_dataset,
                self.dev_batch_size,
                spectrogram_slice_size=self.dev_spectrogram_slice_size,
                spectrogram_slice_pad=self.model.padding,
                **loader_kwargs)
        data_loader = self._train_loader if train else self._dev_loader

        if not train:
            self.exponential_moving_parameter_average.apply_shadow()

        for i, batch in enumerate(data_loader):
            with torch.set_grad_enabled(train):
                predicted_signal = self.model(
                    batch.spectrogram, spectrogram_mask=batch.spectrogram_mask, pad_input=False)
                self._do_loss_and_maybe_backwards(batch, predicted_signal, do_backwards=train)

            # NOTE: This metric should be a positive integer indicating that the `data_loader`
            # is loading faster than the data is getting ingested; otherwise, the `data_loader`
            # is bottlenecking training by loading too slowly.
            if hasattr(data_loader.iterator, '_data_queue'):
                self.metrics['data_queue_size'].update(data_loader.iterator._data_queue.qsize())

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
        assert predicted_signal.shape == batch.signal_mask.shape

        batch_size = predicted_signal.shape[0]
        total_spectrogram_loss = torch.tensor(0.0, device=predicted_signal.device)
        total_generator_loss = torch.tensor(0.0, device=predicted_signal.device)
        total_discriminator_loss = torch.tensor(0.0, device=predicted_signal.device)
        total_discriminator_accuracy = torch.tensor(0.0, device=predicted_signal.device)
        for criterion in self.criterions:
            # NOTE: Even though the signal is zero padded, we can safely ignore it with the
            # assumption that the training examples this model is learning generally start and end
            # with quiet.
            (spectrogram_loss, generator_loss, discriminator_loss,
             discriminator_accuracy) = criterion(
                 predicted_signal,
                 batch.target_signal,
                 comet_ml=self.comet_ml if log_figure else None,
                 do_backwards=do_backwards)

            total_spectrogram_loss += spectrogram_loss / len(self.criterions)
            total_generator_loss += generator_loss / len(self.criterions)
            total_discriminator_loss += discriminator_loss / len(self.criterions)
            total_discriminator_accuracy += discriminator_accuracy / len(self.criterions)

            # TODO: Ensure that this loss reported to Comet is invariant to the slice size
            # so that we can compare accross slice sizes. We can do that by testing if this loss
            # is infact proportional to the slice size (ignoring the boundary undersampling).
            self.metrics['db_mel_%d_spectrogram_magnitude_loss' % criterion.fft_length].update(
                spectrogram_loss, batch_size)
            self.metrics['%d_spectrogram_generator_loss' % criterion.fft_length].update(
                total_generator_loss, batch_size)
            self.metrics['%d_spectrogram_discriminator_loss' % criterion.fft_length].update(
                discriminator_loss, batch_size)
            self.metrics['%d_spectrogram_discriminator_accuracy' % criterion.fft_length].update(
                discriminator_accuracy, batch_size * 2)

        if do_backwards:
            self.optimizer.zero_grad()
            (total_generator_loss + total_spectrogram_loss).backward()
            self.optimizer.step(comet_ml=self.comet_ml)
            self.exponential_moving_parameter_average.update()

        self.metrics['db_mel_spectrogram_magnitude_loss'].update(total_spectrogram_loss, batch_size)
        self.metrics['spectrogram_generator_loss'].update(total_generator_loss, batch_size)
        self.metrics['spectrogram_discriminator_loss'].update(total_discriminator_loss, batch_size)
        self.metrics['spectrogram_discriminator_accuracy'].update(total_discriminator_accuracy,
                                                                  batch_size * 2)

        return total_spectrogram_loss, total_generator_loss, batch.signal_mask.sum()

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
        example = random.sample(self.dev_dataset, 1)[0]
        spectrogram = example.predicted_spectrogram if self.use_predicted else example.spectrogram
        spectrogram = maybe_load_tensor(spectrogram)  # [num_frames, frame_channels]
        target_signal = integer_to_floating_point_pcm(maybe_load_tensor(
            example.spectrogram_audio)).to(self.device)  # [signal_length]
        spectrogram = spectrogram.to(self.device)

        logger.info('Running inference on %d spectrogram frames with %d threads.',
                    spectrogram.shape[0], torch.get_num_threads())

        with evaluate(model, device=self.device):
            self.exponential_moving_parameter_average.apply_shadow()
            predicted = generate_waveform(model, spectrogram, generator=False)
            self.exponential_moving_parameter_average.restore()

        total_spectrogram_loss = torch.tensor(0.0, device=self.device)
        total_generator_loss = torch.tensor(0.0, device=self.device)
        total_discriminator_loss = torch.tensor(0.0, device=self.device)
        total_discriminator_accuracy = torch.tensor(0.0, device=self.device)
        for criterion in self.criterions:
            (spectrogram_loss, generator_loss, discriminator_loss,
             discriminator_accuracy) = criterion(
                 predicted, target_signal, comet_ml=self.comet_ml)

            total_spectrogram_loss += spectrogram_loss / len(self.criterions)
            total_generator_loss += generator_loss / len(self.criterions)
            total_discriminator_loss += discriminator_loss / len(self.criterions)
            total_discriminator_accuracy += discriminator_accuracy / len(self.criterions)

            self.comet_ml.log_metrics({
                'single/%d_spectrogram_generator_loss' % criterion.fft_length:
                    generator_loss.item(),
                'single/%d_spectrogram_discriminator_loss' % criterion.fft_length:
                    discriminator_loss.item(),
                'single/%d_spectrogram_discriminator_accuracy' % criterion.fft_length:
                    discriminator_accuracy.item(),
                'single/db_mel_%d_spectrogram_magnitude_loss' % criterion.fft_length:
                    spectrogram_loss.item()
            })

        self.comet_ml.log_metrics({
            'single/db_mel_spectrogram_magnitude_loss': total_spectrogram_loss.item(),
            'single/spectrogram_generator_loss': total_generator_loss.item(),
            'single/spectrogram_discriminator_loss': total_discriminator_loss.item(),
            'single/spectrogram_discriminator_accuracy': total_discriminator_accuracy.item()
        })
        self.comet_ml.log_audio(
            audio={
                'gold_audio': target_signal,
                'predicted_audio': predicted,
            },
            tag=self.DEV_INFERRED_LABEL,
            text=example.text,
            speaker=str(example.speaker),
            db_mel_spectrogram_magnitude_loss=total_spectrogram_loss.item())
        self.comet_ml.log_figure('input_spectrogram', plot_spectrogram(spectrogram.detach().cpu()))

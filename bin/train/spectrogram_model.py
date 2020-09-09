from collections import defaultdict
from functools import partial
from itertools import chain
from pathlib import Path

import argparse
import logging
import math
import random
import sqlite3
import typing

# NOTE: Needs to be imported before torch
import comet_ml  # noqa

from hparams import add_config
from hparams import get_config
from hparams import HParams
from hparams import parse_hparam_args
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data.dataloader import _MultiProcessingDataLoaderIter  # type: ignore
from torchnlp.utils import get_total_parameters
from torchnlp.utils import lengths_to_mask
from torchnlp.utils import tensors_to

import torch

from src.bin import _config
from src.bin._utils import batch_spectrogram_examples
from src.bin._utils import connect
from src.bin._utils import Dataset
from src.bin._utils import fetch_phonemes
from src.bin._utils import fetch_texts
from src.bin._utils import get_rms_level
from src.bin._utils import get_spectrogram_example
from src.bin._utils import handle_null_alignments
from src.bin._utils import init_distributed
from src.bin._utils import maybe_make_experiment_directories
from src.bin._utils import model_context
from src.bin._utils import SpectrogramModelCheckpoint
from src.bin._utils import SpectrogramModelExample
from src.bin._utils import SpectrogramModelExampleBatch
from src.bin._utils import update_audio_file_metadata
from src.bin._utils import update_word_representations
from src.bin._utils import worker_init_fn
from src.utils import dict_collapse
from src.utils import DistributedAveragedMetric
from src.utils import get_weighted_stdev
from src.utils import load
from src.utils import load_most_recent_file
from src.utils import save

import src

logger = logging.getLogger(__name__)


def _configure(hparams: typing.Dict[str, typing.Any],
               comet_ml: typing.Union[comet_ml.Experiment, comet_ml.ExistingExperiment]):
    _config.configure_audio_processing()
    _config.configure_models()

    train_batch_size = 56
    add_config({
        'src.bin.train.spectrogram_model': {
            'trainer.Trainer.__init__':
                HParams(
                    lr_multiplier_schedule=partial(
                        src.optimizers.warmup_lr_multiplier_schedule, warmup=500),
                    # SOURCE: Tacotron 2
                    # To train the feature prediction network, we apply the standard
                    # maximum-likelihood training procedure (feeding in the correct
                    # output instead of the predicted output on the decoder side, also
                    # referred to as teacher-forcing) with a batch size of 64 on a
                    # single GPU.
                    # NOTE: Parameters set after experimentation on a 2
                    # Px100 GPU.
                    train_batch_size=train_batch_size,
                    dev_batch_size=train_batch_size * 4,

                    # SOURCE (Tacotron 2):
                    # We use the Adam optimizer [29] with β1 = 0.9, β2 = 0.999
                    optimizer=torch.optim.Adam,

                    # SOURCE (Tacotron 2 Author):
                    # The author confirmed they used BCE loss in Google Chat.
                    criterion_stop_token=torch.nn.BCEWithLogitsLoss,

                    # SOURCE: Tacotron 2
                    # We minimize the summed mean squared error (MSE) from before and
                    # after the post-net to aid convergence.
                    criterion_spectrogram=torch.nn.MSELoss),
            'trainer.Trainer._do_loss_and_maybe_backwards':
                HParams(
                    # NOTE: The loss is calibrated to match the loss computed on a
                    # Tacotron-2 spectrogram input.
                    spectrogram_loss_scalar=1 / 100,
                    # NOTE: This stop token loss is calibrated to prevent overfitting
                    # on the stop token before the model is able to model the
                    # spectrogram.
                    stop_token_loss_scalar=1.0,

                    # NOTE: This parameter is based on https://arxiv.org/abs/2002.08709.
                    # We set a loss cut off to prevent overfitting.
                    # TODO: Try increasing the stop token minimum loss because it still
                    # overfit.
                    stop_token_minimum_loss=0.0105),
            'data_loader.get_normalized_half_gaussian':
                HParams(
                    # NOTE: We approximated the uncertainty in the stop token by viewing
                    # the stop token predictions by a fully trained model without
                    # this smoothing. We found that a fully trained model would
                    # learn a similar curve over 4 - 8 frames in January 2020, on Comet.
                    # NOTE: This was rounded up to 10 after the spectrograms got
                    # 17% larger.
                    # TODO: In July 2020, the spectrogram size was decreased by 2x, we
                    # should test decreasing `length` by 2x, also.
                    length=10,
                    standard_deviation=2),
        }
    })

    add_config({'src.environment.set_seed': HParams(seed=_config.RANDOM_SEED)})

    add_config({
        # SOURCE (Tacotron 2):
        # We use the Adam optimizer [29] with β1 = 0.9, β2 = 0.999, eps = 10−6
        # learning rate of 10−3
        # We also apply L2 regularization with weight 10−6
        # NOTE: An approach without L2 regularization was beneficial based on Comet experiments
        # in March 2020.
        'torch.optim.adam.Adam.__init__':
            HParams(eps=10**-6, weight_decay=0, lr=10**-3, amsgrad=True, betas=(0.9, 0.999))
    })

    add_config(hparams)
    src.environment.set_seed()

    comet_ml.log_parameters(dict_collapse(get_config()))


def _get_dataset(
    comet_ml: typing.Union[comet_ml.Experiment, comet_ml.ExistingExperiment]
) -> typing.Tuple[Dataset, Dataset]:
    """ Get text-to-speech dataset for training and evaluation. """
    train_dataset, dev_dataset = _config.get_dataset()
    comet_ml.log_parameters({
        'num_train_examples': sum([len(v) for v in train_dataset.values()]),
        'num_dev_examples': sum([len(v) for v in dev_dataset.values()]),
        'length_train_text': sum([sum([len(e.text) for e in v]) for v in train_dataset.values()]),
        'length_dev_text': sum([sum([len(e.text) for e in v]) for v in dev_dataset.values()]),
    })
    return train_dataset, dev_dataset


class _IterableDataset(torch.utils.data.IterableDataset):

    def __init__(
        self,
        dataset: Dataset,
        bucket_size: int,
        connection: sqlite3.Connection,
        input_encoder: src.spectrogram_model.InputEncoder,
        comet_ml: typing.Union[comet_ml.Experiment, comet_ml.ExistingExperiment],
    ):
        """ `torch.utils.data.IterableDataset` that's compatible with `torch.utils.data.DataLoader`.
        Additionally, this object captures statistics on the dataset generated.

        Args:
            dataset
            bucket_size: A batch of examples is sampled and sorted to minimize padding.
            connection
            input_encoder
            comet_ml
        """
        super().__init__()
        self.dataset = dataset
        self.bucket_size = bucket_size
        self.total_spectrogram_frames = 0
        self.total_spectrograms = 0
        self.speaker_total_spectrogram_frames: typing.Dict[src.datasets.Speaker,
                                                           int] = defaultdict(int)
        self.comet_ml = comet_ml
        self.connection = connection
        self.input_encoder = input_encoder
        # The issue with exposing the generator, is it feels hacky. I'd rather have a generator
        # create seperately? I don't mind the idea of also just exposing more data from our pipeline
        # ... That means that we can add better logging for training and other parts of the
        # pipeline.

    @property
    def average_spectrogram_length(self) -> float:
        return self.total_spectrogram_frames / self.total_spectrograms

    def __iter__(self) -> typing.Generator[SpectrogramModelExample, None, None]:
        generator = _config.get_dataset_generator(self.dataset)
        while True:
            examples = []
            for _ in range(self.bucket_size):
                example = next(generator)
                preprocessed_example = get_spectrogram_example(example, self.connection,
                                                               self.input_encoder)
                examples.append(preprocessed_example)

                # Extract statistics from the generated data.
                num_spectrogram_frames = examples[-1].spectrogram.shape[0]
                self.total_spectrogram_frames += num_spectrogram_frames
                self.speaker_total_spectrogram_frames[example.speaker] += num_spectrogram_frames
                self.total_spectrograms += 1
                for speaker, count in self.speaker_total_spectrogram_frames.items():
                    label = src.environment.text_to_label(speaker.name)
                    percent = count / self.total_spectrogram_frames
                    comet_ml.log_metric(f"dataset/${label}/percent_spectrogram_frames", percent)

            yield from sorted(examples, key=lambda e: e.spectrogram.shape[0])


def _get_data_loader(dataset: _IterableDataset, batch_size: int, device_index: int,
                     **kwargs) -> torch.utils.data.DataLoader:
    """ Initialize a data loader for `dataset`. """
    return torch.utils.data.DataLoader(
        dataset,
        pin_memory=True,
        batch_size=batch_size,
        worker_init_fn=partial(worker_init_fn, seed=_config.RANDOM_SEED, device_index=device_index),
        collate_fn=batch_spectrogram_examples,
        **kwargs)


def _get_data_loaders(
    device: torch.device,
    train_dataset: Dataset,
    dev_dataset: Dataset,
    input_encoder: src.spectrogram_model.InputEncoder,
    connection: sqlite3.Connection,
    comet_ml: typing.Union[comet_ml.Experiment, comet_ml.ExistingExperiment],
    train_batch_size: int = 64,
    dev_batch_size: int = 256,
    bucket_size_multiplier: int = 10,
    num_data_loader_workers: int = 8
) -> typing.Tuple[_IterableDataset, _IterableDataset, _MultiProcessingDataLoaderIter,
                  _MultiProcessingDataLoaderIter]:
    """ Initialize training and development data loaders.  """
    bucket_size = bucket_size_multiplier * train_batch_size
    DataLoaderPartial = partial(
        _get_data_loader, num_workers=num_data_loader_workers, device_index=device.index)
    _IterableDatasetPartial = partial(
        _IterableDataset,
        connection=connection,
        input_encoder=input_encoder,
        comet_ml=comet_ml,
        bucket_size=bucket_size)
    train_dataset_iter = _IterableDatasetPartial(train_dataset)
    dev_dataset_iter = _IterableDatasetPartial(dev_dataset)
    iter_loader = lambda d: (tensors_to(b, device=device, non_blocking=True) for b in d)
    train_data_loader = iter_loader(DataLoaderPartial(train_dataset_iter, train_batch_size))
    dev_data_loader = iter_loader(DataLoaderPartial(dev_dataset_iter, dev_batch_size))
    return train_dataset_iter, dev_dataset_iter, train_data_loader, dev_data_loader


def _get_input_encoder(
    speakers: typing.List[src.datasets.Speaker],
    connection: sqlite3.Connection,
    comet_ml: typing.Union[comet_ml.Experiment, comet_ml.ExistingExperiment],
) -> src.spectrogram_model.InputEncoder:
    """ Initialize an input encoder to encode model input. """
    input_encoder = src.spectrogram_model.InputEncoder(
        fetch_texts(connection), fetch_phonemes(connection), _config.PHONEME_SEPARATOR, speakers)
    comet_ml.log_parameters({
        'grapheme_vocab_size': input_encoder.grapheme_encoder.vocab_size,
        'grapheme_vocab': sorted(input_encoder.grapheme_encoder.vocab),
        'phoneme_vocab_size': input_encoder.phoneme_encoder.vocab_size,
        'phoneme_vocab': sorted(input_encoder.phoneme_encoder.vocab),
        'num_speakers': input_encoder.speaker_encoder.vocab_size,
        'speakers': sorted(input_encoder.speaker_encoder.vocab)
    })
    return input_encoder


def _get_model(
    device: torch.device,
    comet_ml: typing.Union[comet_ml.Experiment, comet_ml.ExistingExperiment],
) -> torch.nn.Module:
    """ Initialize a model for training. """
    model = src.spectrogram_model.SpectrogramModel().to(device)
    # Learn more about `DistributedDataParallel` here:
    # https://discuss.pytorch.org/t/proper-distributeddataparallel-usage/74564
    comet_ml.set_model_graph(str(model))
    comet_ml.log_parameters({'num_parameter': get_total_parameters(model)})
    return DistributedDataParallel(model, device_ids=[device], output_device=device)


def _get_optimizers(
    model: torch.nn.Module,
    device: torch.device,
    optimizer: typing.Type[torch.optim.adam.Adam] = torch.optim.adam.Adam,
    lr_multiplier_schedule: typing.Callable[[int], float] = partial(
        src.optimizers.warmup_lr_multiplier_schedule, warmup=500),
) -> typing.Tuple[src.optimizers.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
    """ Initialize optimization modules that update model parameters based on their gradients. """
    torch_optim = optimizer(params=filter(lambda p: p.requires_grad, model.parameters()))
    optim = src.optimizers.AutoOptimizer(torch_optim).to(device)
    scheduler = LambdaLR(optim.optimizer, lr_multiplier_schedule)
    return optim, scheduler


class _Metrics(typing.NamedTuple):
    """ Initialize various training metrics to track in this distributed environment.

    Args:
        attention_norm: This measures the average p-norm of attention alignments; therefore,
            this metric goes to 1.0 the large the attention alignments are and it goes to zero
            the smaller the attention alignments are.
        attention_std: This measures the average discrete standard deviation of attention
            alignments; therefore, this metric goes to zero the less scatted the alignment is.
        average_predicted_rms_level: This measures the average loudess of the predicted
            spectrograms.
        average_relative_speed: This measures the average number of frames per
        average_target_rms_level: This measures the total length of all predicted audio over the
            total length of ground truth audio. In effect, it measures the model's average speed
            relative to the original lengths.
        data_queue_size: This measures the data loader queue. This metric should be a positive
            integer indicating that the `data_loader` is loading faster than the data is getting
            ingested; otherwise, the `data_loader` is bottlenecking training by loading too slowly.
        reached_max_frames: The measures the number of predicted spectrograms have reached the
            threshold for maximum length.
        spectrogram_loss: This measures the average difference between the original and predicted
            spectrogram.
        stop_token_accuracy: This measures the accuracy of the stop token.
        stop_token_loss: This measures the average difference between the original and predicted
            stop token distribution.
    """
    attention_norm: DistributedAveragedMetric = DistributedAveragedMetric()
    attention_std: DistributedAveragedMetric = DistributedAveragedMetric()
    average_predicted_rms_level: DistributedAveragedMetric = DistributedAveragedMetric()
    average_relative_speed: DistributedAveragedMetric = DistributedAveragedMetric()
    average_target_rms_level: DistributedAveragedMetric = DistributedAveragedMetric()
    data_queue_size: DistributedAveragedMetric = DistributedAveragedMetric()
    reached_max_frames: DistributedAveragedMetric = DistributedAveragedMetric()
    spectrogram_loss: DistributedAveragedMetric = DistributedAveragedMetric()
    stop_token_accuracy: DistributedAveragedMetric = DistributedAveragedMetric()
    stop_token_loss: DistributedAveragedMetric = DistributedAveragedMetric()


def _update_attention_metrics(metrics: _Metrics,
                              alignments: torch.Tensor,
                              mask: torch.Tensor,
                              norm: float = math.inf):
    """ Update `attention_norm` and `attention_std` in `metrics` given `alignments`.

    Args:
        metrics
        alignments (torch.FloatTensor [num_frames, batch_size, num_tokens])
        mask (torch.BoolTensor [num_frames, batch_size])
        norm
    """
    if mask.sum() != 0:
        assert len(alignments.shape) == 3
        assert len(mask.shape) == 2
        metrics.attention_norm.update(alignments.norm(norm, dim=2).masked_select(mask), mask.sum())
        metrics.attention_std.update(get_weighted_stdev(alignments, dim=2, mask=mask), mask.sum())


def _log_metrics(
    metrics: _Metrics,
    comet_ml: typing.Union[comet_ml.Experiment, comet_ml.ExistingExperiment],
    prefix: str,
    get_metric: typing.Callable[[DistributedAveragedMetric], float],
):
    """ Log `metrics` to `comet_ml` under the `prefix`.
    """
    for field in metrics._fields:
        metric = get_metric(getattr(metrics, field))
        if metric is not None:
            if field == 'average_target_rms_level' or field == 'average_predicted_rms_level':
                metric = src.audio.power_to_db(torch.tensor(metric)).item()
            comet_ml.log_metric(f"${prefix}/${field}", metric)
    comet_ml.log_metric(
        f"${prefix}/average_loudness_delta",
        src.audio.power_to_db(torch.tensor(metrics.average_predicted_rms_level)) -
        src.audio.power_to_db(torch.tensor(metrics.average_target_rms_level)))


def _visualize_source_vs_target(
    comet_ml: typing.Union[comet_ml.Experiment, comet_ml.ExistingExperiment],
    batch: SpectrogramModelExampleBatch,
    predicted_spectrogram: torch.Tensor,
    predicted_stop_token: torch.Tensor,
    predicted_alignments: torch.Tensor,
):
    """ Visualize predictions as compared to the original `batch`.

    Args:
        comet_ml
        batch
        predicted_spectrogram (torch.FloatTensor [num_frames, batch_size, frame_channels]):
            Spectrogram frames.
        predicted_stop_token (torch.FloatTensor [num_frames, batch_size]): Stopping probability for
            each frame.
        predicted_alignments (torch.FloatTensor [num_frames, batch_size, num_tokens]): Attention
            alignment between `frames` and `tokens`.
    """
    batch_size = predicted_spectrogram.shape[1]
    item = random.randint(0, batch_size - 1)
    spectrogram_length = int(batch.spectrogram.lengths[0, item].item())
    text_length = int(batch.encoded_text.lengths[0, item].item())
    # spectrogram [num_frames, frame_channels]
    predicted_spectrogram = predicted_spectrogram[:spectrogram_length, item]
    # gold_spectrogram [num_frames, frame_channels]
    gold_spectrogram = batch.spectrogram.tensor[:spectrogram_length, item]
    predicted_delta = abs(gold_spectrogram - predicted_spectrogram)
    predicted_alignments = predicted_alignments[:spectrogram_length, item, :text_length]
    stop_token_plot = src.visualize.plot_stop_token(predicted_stop_token[:spectrogram_length, item])
    comet_ml.log_figures({
        'delta_spectrogram': src.visualize.plot_mel_spectrogram(predicted_delta),
        'gold_spectrogram': src.visualize.plot_mel_spectrogram(gold_spectrogram),
        'predicted_spectrogram': src.visualize.plot_mel_spectrogram(predicted_spectrogram),
        'alignment': src.visualize.plot_attention(predicted_alignments),
        'stop_token': stop_token_plot,
    })


def _run_step(step: int,
              batch: SpectrogramModelExampleBatch,
              data_loader: _MultiProcessingDataLoaderIter,
              dataset_iterable: _IterableDataset,
              model: torch.nn.Module,
              optimizer: src.optimizers.Optimizer,
              scheduler: torch.optim.lr_scheduler.LambdaLR,
              metrics: _Metrics,
              comet_ml: typing.Union[comet_ml.Experiment, comet_ml.ExistingExperiment],
              visualize: bool = False,
              spectrogram_loss_scalar: float = 1 / 100,
              stop_token_min_loss: float = 0.0105) -> int:
    """ Update the model using the next batch from `train_data_loader`.

    Args:
        step
        batch
        data_loader
        dataset_iterable
        model
        optimizer
        scheduler
        metrics
        comet_ml
        visualize: If `True` visualize the results with `comet_ml`.
        spectrogram_loss_scalar: This scales the spectrogram loss by some value.
        stop_token_min_loss: This thresholds the stop token loss to prevent overfitting.

    Returns:
        step
    """
    frames, stop_token, alignment, spectrogram_loss, stop_token_loss = model(
        tokens=batch.encoded_text.tensor,
        speaker=batch.encoded_speaker.tensor,
        target_frames=batch.spectrogram.tensor,
        target_stop_token=batch.stop_token.tensor,
        num_tokens=batch.encoded_text.lengths,
        target_lengths=batch.spectrogram.lengths)

    if model.training:  # Update model weights
        optimizer.zero_grad()

        # NOTE: We sum over the `num_frames` dimension to ensure that we don't bias based on
        # `num_frames`. For example, a larger `num_frames` means that the denominator is larger;
        # therefore, the loss value for each element is smaller.
        # NOTE: We average accross `batch_size` and `frame_channels` so that the loss magnitude is
        # invariant to those variables.

        # spectrogram_loss [num_frames, batch_size, frame_channels] → [1]
        # stop_token_loss [num_frames, batch_size] → [1]
        average_spectrogram_length = dataset_iterable.average_spectrogram_length
        loss = (spectrogram_loss.sum(dim=0) /
                average_spectrogram_length).mean() * spectrogram_loss_scalar
        loss += ((stop_token_loss.sum(dim=0) / average_spectrogram_length).mean() -
                 stop_token_min_loss).abs() + stop_token_min_loss
        optimizer.step(comet_ml=comet_ml)
        comet_ml.set_step(step)
        scheduler.step()
        step += 1

    if visualize:
        _visualize_source_vs_target(comet_ml, batch, frames, stop_token, alignment)

    # Update metrics, and log those updates.
    num_frames = batch.spectrogram_mask.tensor.sum()
    stop_threshold = model.stop_threshold
    expected_stop_token = (batch.stop_token.tensor > stop_threshold)
    expected_stop_token = expected_stop_token.masked_select(batch.spectrogram_mask.tensor > 0)
    predicted_stop_token = (torch.sigmoid(stop_token) > stop_threshold)
    predicted_stop_token = predicted_stop_token.masked_select(batch.spectrogram_mask.tensor > 0)
    stop_token_accuracy = (expected_stop_token == predicted_stop_token).float().mean()
    metrics.stop_token_accuracy.update(stop_token_accuracy, num_frames)
    metrics.spectrogram_loss.update(spectrogram_loss.mean(),
                                    batch.spectrogram_extended_mask.tensor.sum())
    metrics.spectrogram_loss.update(stop_token_loss.mean(), num_frames)
    _update_attention_metrics(metrics, alignment, batch.spectrogram_mask)
    metrics.data_queue_size.update(data_loader._data_queue.qsize())
    _log_metrics(metrics, comet_ml, 'step/', lambda m: m.sync().last_update())

    return step


def _update_rms_level_metrics(metrics,
                              target: torch.Tensor,
                              predicted: torch.Tensor,
                              target_mask: typing.Optional[torch.Tensor] = None,
                              predicted_mask: typing.Optional[torch.Tensor] = None,
                              **kwargs):
    """ Update `average_target_rms_level` and `average_predicted_rms_level` in `metrics`.

    Args:
        target (torch.FloatTensor [num_frames, batch_size, frame_channels]): Target spectrogram.
        predicted (torch.FloatTensor [num_frames, batch_size, frame_channels]): Predicted
            spectrogram.
        target_mask (torch.FloatTensor [num_frames, batch_size]): Target spectrogram mask.
        predicted_mask (torch.FloatTensor [num_frames, batch_size]): Predicted spectrogram mask.
        **kwargs: Additional key word arguments passed to `self._get_loudness`.
    """
    target_rms = src.audio.db_to_power(get_rms_level(target, target_mask, **kwargs))
    predicted_rms = src.audio.db_to_power(get_rms_level(predicted, predicted_mask, **kwargs))
    metrics.average_target_rms_level.update(
        target_rms,
        target.numel() if target_mask is None else target_mask.sum())
    metrics.average_predicted_rms_level.update(
        predicted_rms,
        predicted_rms.numel() if predicted_mask is None else predicted_mask.sum())


def _visualize_inferred(
    comet_ml: typing.Union[comet_ml.Experiment, comet_ml.ExistingExperiment],
    batch: SpectrogramModelExampleBatch,
    predicted_spectrogram: torch.Tensor,
    predicted_stop_token: torch.Tensor,
    predicted_alignments: torch.Tensor,
):
    """ Run in inference mode and visualize results.

    Args:
        comet_ml
        batch
        predicted_spectrogram (torch.FloatTensor [num_frames, batch_size, frame_channels]):
            Spectrogram frames.
        predicted_stop_token (torch.FloatTensor [num_frames, batch_size]): Stopping probability for
            each frame.
        predicted_alignments (torch.FloatTensor [num_frames, batch_size, num_tokens]): Attention
            alignment between `frames` and `tokens`.
    """
    batch_size = predicted_spectrogram.shape[1]
    item = random.randint(0, batch_size - 1)
    spectrogram_length = int(batch.spectrogram.lengths[0, item].item())
    text_length = int(batch.encoded_text.lengths[0, item].item())
    # spectrogram [num_frames, frame_channels]
    predicted_spectrogram = predicted_spectrogram[:spectrogram_length, item]
    # gold_spectrogram [num_frames, frame_channels]
    gold_spectrogram = batch.spectrogram.tensor[:spectrogram_length, item]
    predicted_loudness = get_rms_level(predicted_spectrogram.unsqueeze(1))
    gold_loudness = get_rms_level(gold_spectrogram.unsqueeze(1))
    predicted_alignments = predicted_alignments[:spectrogram_length, item, :text_length]
    stop_token_plot = src.visualize.plot_stop_token(predicted_stop_token[:spectrogram_length, item])
    audio = {
        'predicted_griffin_lim_audio': src.audio.griffin_lim(predicted_spectrogram.cpu().numpy()),
        'gold_griffin_lim_audio': src.audio.griffin_lim(gold_spectrogram.numpy()),
        'gold_audio': batch.audio[item].numpy(),
    }
    comet_ml.log_audio(
        audio=audio,
        context=comet_ml.context,
        text=batch.text[item],
        speaker=batch.speaker[item],
        predicted_loudness=predicted_loudness,
        gold_loudness=gold_loudness)
    comet_ml.log_figures({
        'gold_spectrogram': src.visualize.plot_mel_spectrogram(gold_spectrogram),
        'predicted_spectrogram': src.visualize.plot_mel_spectrogram(predicted_spectrogram),
        'alignment': src.visualize.plot_attention(predicted_alignments),
        'stop_token': stop_token_plot,
    })


def _evaluate_inference(
    batch: SpectrogramModelExampleBatch,
    model: torch.nn.Module,
    metrics: _Metrics,
    comet_ml: typing.Union[comet_ml.Experiment, comet_ml.ExistingExperiment],
    visualize: bool = False,
):
    """ Run the model in inference mode, and measure it's results.

    Args:
        batch
        model
        metrics
        comet_ml
        visualize: If `True` visualize the results with `comet_ml`.
    """
    # NOTE: Remove predictions that diverged (reached max) as to not skew other
    # metrics. We count these sequences seperatly with `reached_max_frames`.
    frames, stop_tokens, alignments, lengths, reached_max = model(
        batch.encoded_text.tensor,
        batch.encoded_speaker.tensor,
        batch.encoded_text.lengths,
        filter_reached_max=True,
        mode='infer')

    if visualize:  # TODO: Visualize results that overflowed, also.
        _visualize_inferred(comet_ml, batch, frames, stop_tokens, alignments)

    if lengths.numel() > 0:
        # NOTE: `average_relative_speed` computes the total length of all predicted
        # audio over the total length of ground truth audio. In effect, it measures
        # the model's average speed relative to the original lengths.
        reached_max_filter = ~reached_max.squeeze()
        batch_lengths = batch.spectrogram.lengths[:, reached_max_filter]
        metrics.average_relative_speed.update(lengths.sum().float() / batch_lengths.sum().float(),
                                              lengths.sum())

        device = lengths.device
        mask = lengths_to_mask(lengths, device=device).transpose(0, 1)
        _update_rms_level_metrics(batch.spectrogram.tensor[:, reached_max_filter], frames,
                                  batch.spectrogram_mask.tensor[:, reached_max_filter], mask)
        _update_attention_metrics(metrics, alignments, mask)

    metrics.reached_max_frames.update(reached_max.float().mean(), reached_max.numel())


def _run_worker(device_index: int,
                run_root: Path,
                checkpoints_directory: Path,
                checkpoint: typing.Optional[SpectrogramModelCheckpoint],
                train_dataset: Dataset,
                dev_dataset: Dataset,
                comet_ml_partial: typing.Callable[[..., typing.Any],
                                                  typing.Union[comet_ml.Experiment,
                                                               comet_ml.ExistingExperiment]],
                hparams: typing.Dict[str, typing.Any],
                num_steps_per_epoch: int = 1000):
    """ Loop for training and periodically evaluating the model.

    TODO:
        - Incorperate checkpointing
        - Updated `SpectrogramModel` to accept new arguments
        - Consider moving comet to the global scope via environment variables.
        - Part of what helped us develop the signal model is determinism... If we remove
          that determinisim it'll make it difficult to evaluate. At least for inference, we might
          be able to still be deterministic? We don't need to use a data loader... maybe. Never
          mind. We'd still need to get loudness annotations, etc. We could develop a test set...
          that's preloaded?
    """
    # Setup worker logging, visualization and environment
    src.environment.set_basic_logging_config(device_index)
    device = init_distributed(device_index)
    connection = connect(_config.TRAINING_DB)
    comet_ml = comet_ml_partial(disabled=not src.distributed.is_master(), auto_output_logging=False)
    _configure(hparams, comet_ml)

    # Initialize modules and variables for training
    if checkpoint is None:
        input_encoder = _get_input_encoder(list(train_dataset.keys()), connection, comet_ml)
        model = _get_model(device, comet_ml)
        optimizer, scheduler = _get_optimizers(model, device)
        step = 0
    else:
        input_encoder = checkpoint.input_encoder
        model = checkpoint.model
        optimizer = checkpoint.optimizer
        scheduler = checkpoint.scheduler
        step = checkpoint.step
    train_dataset_iter, dev_dataset_iter, train_data_loader, dev_data_loader = _get_data_loaders(
        device, train_dataset, dev_dataset, input_encoder, connection, comet_ml)
    metrics = _Metrics()

    # Run training loop
    _step = partial(
        _run_step,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        metrics=metrics,
        comet_ml=comet_ml)
    _model_context = partial(model_context, model=model, comet_ml=comet_ml)
    while True:
        epoch = step // num_steps_per_epoch
        logger.info('Running Epoch %d, Step %d', epoch, step)
        comet_ml.log_current_epoch(epoch)

        with _model_context(name='training', is_train=True):
            for i, batch in zip(range(num_steps_per_epoch), train_data_loader):
                step = _step(step, batch, train_data_loader, train_dataset_iter, visualize=i == 0)
            _log_metrics(metrics, comet_ml, 'epoch/', lambda m: m.sync().reset())

        with _model_context(name='evaluation', is_train=False):
            for i, batch in zip(range(num_steps_per_epoch), dev_data_loader):
                _step(step, batch, dev_data_loader, dev_dataset_iter, visualize=i == 0)
            _log_metrics(metrics, comet_ml, 'epoch/', lambda m: m.sync().reset())

        with _model_context(name='inference_evaluation', is_train=False):
            for i, batch in zip(range(num_steps_per_epoch), dev_data_loader):
                _evaluate_inference(batch, model, metrics, comet_ml, visualize=i == 0)
            _log_metrics(metrics, comet_ml, 'epoch/', lambda m: m.sync().reset())

        comet_ml.log_epoch_end(epoch)

        save(
            checkpoints_directory / f"step_{step}.pt",
            SpectrogramModelCheckpoint(
                checkpoints_directory=checkpoints_directory,
                comet_ml_experiment_key=comet_ml.get_key(),
                comet_ml_project_name=comet_ml.project_name,
                input_encoder=input_encoder,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                step=step))


def run(experiment_name: typing.Optional[str] = None,
        comet_ml_experiment_key: typing.Optional[str] = None,
        comet_ml_project_name: typing.Optional[str] = None,
        experiment_tags: typing.List[str] = [],
        experiment_root: Path = src.environment.SPECTROGRAM_MODEL_EXPERIMENTS_PATH /
        src.environment.bash_time_label(),
        checkpoint: typing.Optional[SpectrogramModelCheckpoint] = None,
        hparams: typing.Dict[str, typing.Any] = {},
        minimum_disk_space: float = 0.2):
    """ Run the spectrogram model training script.

    TODO: Consider ignoring ``add_tags`` if `checkpoint` is loaded; or consider saving in the
    `checkpoint` the ``name`` and ``tags``; or consider fetching tags from the Comet.ML API.
    """
    # Check setup before training
    src.environment.check_module_versions()  # Check
    src.environment.assert_enough_disk_space(minimum_disk_space)

    # Set up logging, visualization and environment
    src.environment.set_basic_logging_config()
    comet_ml_partial = partial(src.visualize.CometML, project_name=comet_ml_project_name)
    comet_ml = comet_ml_partial(experiment_key=comet_ml_experiment_key)
    comet_ml.set_name(experiment_name)
    comet_ml.add_tags(experiment_tags)
    recorder = src.environment.RecordStandardStreams().start()
    _configure(hparams, comet_ml)
    root_path, checkpoints_path = maybe_make_experiment_directories(experiment_root, recorder,
                                                                    checkpoint)
    comet_ml.log_other('directory', str(root_path))
    connection = connect(_config.TRAINING_DB)

    # Load dataset
    train_dataset, dev_dataset = _get_dataset(comet_ml)

    # Preprocess dataset
    examples = list(chain(*tuple(chain(train_dataset.values(), dev_dataset.values()))))
    update_audio_file_metadata(connection, [e.audio_path for e in examples])
    train_dataset = handle_null_alignments(connection, train_dataset)
    dev_dataset = handle_null_alignments(connection, dev_dataset)
    update_word_representations(
        connection, [e.text for e in examples], separator=_config.PHONEME_SEPARATOR)

    # Spawn workers
    comet_ml_partial = partial(comet_ml_partial, experiment_key=comet_ml.get_key())
    args = (root_path, checkpoints_path, checkpoint, train_dataset, dev_dataset, comet_ml_partial,
            hparams)
    src.distributed.spawn(_run_worker, args=args)


if __name__ == '__main__':  # pragma: no cover
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--checkpoint',
        const=True,
        type=str,
        default=None,
        action='store',
        nargs='?',
        help='Without a value, this loads the most recent checkpoint; '
        'otherwise, expects a checkpoint file path.')
    parser.add_argument('--name', type=str, default=None, help='Name of the experiment.')
    parser.add_argument(
        '--project_name',
        type=str,
        default=None,
        help='Name of the comet.ml project to store a new experiment in.')
    parser.add_argument('--tags', default=[], nargs='+', help='List of tags for a new experiments.')

    args, unparsed_args = parser.parse_known_args()

    checkpoint: typing.Optional[SpectrogramModelCheckpoint] = None
    if isinstance(args.checkpoint, str):
        checkpoint = typing.cast(SpectrogramModelCheckpoint, load(args.checkpoint))
    elif isinstance(args.checkpoint, bool) and args.checkpoint:
        pattern = src.environment.SPECTROGRAM_MODEL_EXPERIMENTS_PATH / '**/*.pt'
        checkpoint = typing.cast(SpectrogramModelCheckpoint, load_most_recent_file(pattern, load))

    comet_ml_experiment_key = None
    if checkpoint is not None:
        args.project_name = checkpoint.comet_ml_project_name
        comet_ml_experiment_key = checkpoint.comet_ml_experiment_key

    run(experiment_name=args.name,
        experiment_tags=args.tags,
        comet_ml_experiment_key=comet_ml_experiment_key,
        comet_ml_project_name=args.project_name,
        checkpoint=checkpoint,
        hparams=parse_hparam_args(unparsed_args))

# Learn more:
# https://stackoverflow.com/questions/33533148/how-do-i-specify-that-the-return-type-of-a-method-is-the-same-as-the-class-itsel
from __future__ import annotations

import copy
import dataclasses
import math
import random
import typing
from functools import partial

import torch
import torch.utils.data
from torch.nn import functional
from torchnlp.encoders.text import SequenceBatch, stack_and_pad_tensors
from torchnlp.samplers import BucketBatchSampler

import lib
import run
import run._config
import run._utils
from run.train import spectrogram_model


class _Slice(typing.NamedTuple):

    # torch.FloatTensor [num_frames, frame_channels]
    spectrogram: torch.Tensor

    # torch.BoolTensor [num_frames, frame_channels]
    spectrogram_mask: torch.Tensor

    # torch.FloatTensor [signal_length]
    target_signal: torch.Tensor

    # torch.BoolTensor [signal_length]
    signal_mask: torch.Tensor


def _get_slice(
    spectrogram: torch.Tensor,
    signal: torch.Tensor,
    spectrogram_slice_size: int,
    spectrogram_slice_pad: int,
) -> _Slice:
    """Slice the data into bite sized chunks that fit onto GPU memory for training.

    Notes:
        * Frames batch needs to line up with the target signal. Each frame, is used to predict
          the target.

    Args:
        spectrogram (torch.FloatTensor [num_frames, channels])
        signal (torch.FloatTensor [signal_length])
        spectrogram_slice_size: In spectrogram frames, size of slice.
        spectrogram_slice_pad: Pad the spectrogram slice with ``frame_pad`` frames on each side.
    """
    samples, num_frames = signal.shape[0], spectrogram.shape[0]
    samples_per_frame = int(samples / num_frames)
    _make_mask = partial(torch.ones, dtype=torch.bool)

    # Signal model requires that there is a scaling factor between the signal and frames
    assert samples % num_frames == 0

    signal_mask = _make_mask(signal.shape[0], device=signal.device)
    spectrogram_mask = _make_mask(spectrogram.shape[0], device=spectrogram.device)

    # Pad spectrogram and signal
    spectrogram_zeros = spectrogram_slice_size - 1 + spectrogram_slice_pad
    spectrogram = functional.pad(spectrogram, [0, 0, spectrogram_zeros, spectrogram_zeros])
    spectrogram_mask = functional.pad(spectrogram_mask, [spectrogram_zeros, spectrogram_zeros])

    assert spectrogram_mask.shape[0] == spectrogram.shape[0]

    signal_zeros = (spectrogram_slice_size - 1) * samples_per_frame
    target_signal = functional.pad(signal, [signal_zeros, signal_zeros])
    signal_mask = functional.pad(signal_mask, [signal_zeros, signal_zeros])

    # Get a spectrogram slice
    # ``-spectrogram_slice_size + 1, num_frames - 1`` to ensure there is an equal chance to that
    # a sample will be included inside the slice.
    # For example, with signal ``[1, 2, 3]`` and a ``slice_samples`` of 2 you'd get slices of:
    # (1), (1, 2), (2, 3), (3).
    # With each number represented twice.
    start_frame = random.randint(-spectrogram_slice_size + 1, num_frames - 1)
    end_frame = start_frame + spectrogram_slice_size

    # Get slices from the padded tensors
    # Offset start frame and end frame with `spectrogram_zeros` padding. Increase the range
    # size with `spectrogram_slice_pad` on both ends
    spectrogram_slice = slice(
        start_frame + spectrogram_zeros - spectrogram_slice_pad,
        end_frame + spectrogram_zeros + spectrogram_slice_pad,
    )
    spectrogram_mask_slice = spectrogram_mask[spectrogram_slice]
    spectrogram_slice = spectrogram[spectrogram_slice]

    # Change units from frames to signals and offset with `signal_zeros`
    signal_slice = slice(
        start_frame * samples_per_frame + signal_zeros,
        end_frame * samples_per_frame + signal_zeros,
    )
    target_signal_slice = target_signal[signal_slice]
    signal_mask_slice = signal_mask[signal_slice]

    assert target_signal_slice.shape[0] / samples_per_frame == spectrogram_slice_size
    assert spectrogram_slice.shape[0] == spectrogram_slice_size + spectrogram_slice_pad * 2
    assert spectrogram_mask_slice.shape[0] == spectrogram_slice.shape[0]
    assert target_signal_slice.shape == signal_mask_slice.shape

    return _Slice(spectrogram_slice, spectrogram_mask_slice, target_signal_slice, signal_mask_slice)


@dataclasses.dataclass(frozen=True)
class SpectrogramModelBatch(spectrogram_model._worker.Batch):

    # Spectrograms predicted given `batch`.
    # SequenceBatch[torch.FloatTensor [num_frames, batch_size, frame_channels],
    #               torch.LongTensor [1, batch_size])
    predicted_spectrogram: SequenceBatch


@dataclasses.dataclass(frozen=True)
class Batch:
    """Batch of preprocessed `Span` used to training or evaluating the spectrogram model."""

    batch: SpectrogramModelBatch

    # `batch` `indicies` used to generate this batch of slices.
    indicies: typing.List[int]

    # SequenceBatch[torch.FloatTensor [batch_size, num_frames, frame_channels],
    #               torch.LongTensor [batch_size])
    spectrogram: SequenceBatch

    # NOTE: Mask padding with `False`.
    # SequenceBatch[torch.BoolTensor [batch_size, num_frames], torch.LongTensor [batch_size])
    spectrogram_mask: SequenceBatch

    # SequenceBatch[torch.FloatTensor [batch_size, signal_length], torch.LongTensor [batch_size])
    target_signal: SequenceBatch

    # SequenceBatch[torch.BoolTensor [batch_size, signal_length], torch.LongTensor [batch_size])
    signal_mask: SequenceBatch

    def __len__(self):
        return len(self.indicies)

    def pin_memory(self) -> Batch:
        return run.train._utils.apply_to_tensors(self, lambda t: t.pin_memory())


class DataProcessor(torch.utils.data.IterableDataset):
    def __init__(
        self,
        dataset: run._config.Dataset,
        slice_size: int,
        batch_size: int,
        span_bucket_size: int,
        slice_padding: int,
        checkpoint: spectrogram_model._worker.Checkpoint,
    ):
        """
        Args:
            ...
            slice_size: The number of frames per spectrogram slice.
            batch_size: The batch size of `Batch`.
            span_bucket_size: The bucket size of spans to sample `Batch`s from.
            slice_padding: The number of frames of padding on either side of the spectrogram.
            ...
        """
        iterator = run._utils.SpanGenerator(dataset)
        iterator = BucketBatchSampler(iterator, span_bucket_size, False, self._sort_key)
        self.iterator = typing.cast(typing.Iterator[typing.List[lib.datasets.Span]], iterator)
        self.batch_size = batch_size
        self.slice_padding = slice_padding
        self.slice_size = slice_size
        # NOTE: Without `copy.deepcopy`, `multiprocessing` "spawn" throws an error while replicating
        # this object onto another process...
        # `ValueError: bad value(s) in fds_to_keep`
        # Learn more: https://github.com/pytorch/pytorch/issues/35858
        self.spectrogram_model = copy.deepcopy(checkpoint.model.train(mode=False))
        self._slice = partial(
            _get_slice, spectrogram_slice_size=slice_size, spectrogram_slice_pad=slice_padding
        )
        self._stack = partial(stack_and_pad_tensors, dim=0)
        self._make_spectrogram_model_batch = partial(
            spectrogram_model._data.make_batch, input_encoder=checkpoint.input_encoder
        )

    @staticmethod
    def _sort_key(span: lib.datasets.Span):
        return span.audio_length

    def _make_batches(self, spans: typing.List[lib.datasets.Span]) -> typing.Iterable[Batch]:
        """Sample slices from a batch of predicted spectrograms."""
        batch = self._make_spectrogram_model_batch(spans)
        with torch.no_grad():
            spectrograms, _, _ = self.spectrogram_model(
                tokens=batch.encoded_phonemes.tensor,
                speaker=batch.encoded_speaker.tensor,
                target_frames=batch.spectrogram.tensor,
                num_tokens=batch.encoded_phonemes.lengths,
                tokens_mask=batch.encoded_phonemes_mask.tensor,
                target_mask=batch.spectrogram_mask.tensor,
                mode=lib.spectrogram_model.Mode.FORWARD,
            )
        num_frames = batch.spectrogram.lengths.sum().item()
        weights = batch.spectrogram.lengths.view(-1).float()
        num_batches = int(math.floor(num_frames / self.slice_size / self.batch_size))
        get_spectrogram = lambda i: spectrograms[: batch.spectrogram.lengths[:, i], i]
        for _ in range(num_batches):
            indicies = torch.multinomial(weights, self.batch_size, replacement=True).tolist()
            slices = [self._slice(get_spectrogram(i), batch.audio[i]) for i in indicies]
            yield Batch(
                batch=SpectrogramModelBatch(
                    **lib.utils.dataclass_as_dict(batch),
                    predicted_spectrogram=SequenceBatch(spectrograms, batch.spectrogram.lengths),
                ),
                indicies=indicies,
                spectrogram=self._stack([s.spectrogram for s in slices]),
                spectrogram_mask=self._stack([s.spectrogram_mask for s in slices]),
                target_signal=self._stack([s.target_signal for s in slices]),
                signal_mask=self._stack([s.signal_mask for s in slices]),
            )

    def __iter__(self) -> typing.Iterator[Batch]:
        for spans in self.iterator:
            yield from self._make_batches(spans)

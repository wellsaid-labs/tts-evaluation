import dataclasses
import math
import random
import typing
from functools import partial

import torch
import torch.utils.data
from torch.nn import functional
from torchnlp.encoders.text import SequenceBatch, stack_and_pad_tensors

import lib
import run
import run._config
import run._utils
import run.train
from lib.samplers import BucketBatchSampler
from run.train import _utils
from run.train import spectrogram_model as spectrogram_model_module


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
class SpectrogramModelBatch(spectrogram_model_module._worker.Batch):

    # Spectrograms predicted given `batch`.
    # SequenceBatch[torch.FloatTensor [num_frames, batch_size, frame_channels],
    #               torch.LongTensor [1, batch_size])
    predicted_spectrogram: SequenceBatch


@dataclasses.dataclass(frozen=True)
class Batch(_utils.Batch):
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

    # torch.LongTensor [batch_size]
    speaker: torch.Tensor

    # torch.LongTensor [batch_size]
    session: torch.Tensor

    def __len__(self):
        return len(self.indicies)


class DataProcessor(torch.utils.data.IterableDataset):
    def __init__(
        self,
        dataset: run._config.Dataset,
        slice_size: int,
        batch_size: int,
        span_bucket_size: int,
        slice_padding: int,
        spectrogram_model_input_encoder: spectrogram_model_module._worker.InputEncoder,
        spectrogram_model: lib.spectrogram_model.SpectrogramModel,
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
        self.iterator = typing.cast(typing.Iterator[typing.List[run.data._loader.Span]], iterator)
        self.batch_size = batch_size
        self.slice_padding = slice_padding
        self.slice_size = slice_size
        self.spectrogram_model = spectrogram_model
        self.input_encoder = spectrogram_model_input_encoder
        self._slice = partial(
            _get_slice, spectrogram_slice_size=slice_size, spectrogram_slice_pad=slice_padding
        )
        self._stack = partial(stack_and_pad_tensors, dim=0)
        make_batch = spectrogram_model_module._data.make_batch
        self._make_spectrogram_model_batch = partial(
            make_batch, input_encoder=spectrogram_model_input_encoder
        )

    @staticmethod
    def _sort_key(span: run.data._loader.Span):
        return span.audio_length

    def _make_batches(self, spans: typing.List[run.data._loader.Span]) -> typing.Iterable[Batch]:
        """Sample slices from a batch of predicted spectrograms."""
        batch = self._make_spectrogram_model_batch(spans)
        params = lib.spectrogram_model.Params(
            tokens=batch.encoded_phonemes.tensor,
            speaker=batch.encoded_speaker.tensor,
            session=batch.encoded_session.tensor,
            num_tokens=batch.encoded_phonemes.lengths,
            tokens_mask=batch.encoded_phonemes_mask.tensor,
        )
        preds = self.spectrogram_model(
            params=params,
            target_frames=batch.spectrogram.tensor,
            target_mask=batch.spectrogram_mask.tensor,
            mode=lib.spectrogram_model.Mode.FORWARD,
        )
        num_frames = batch.spectrogram.lengths.sum().item()
        weights = batch.spectrogram.lengths.view(-1).float()
        num_batches = int(math.floor(num_frames / self.slice_size / self.batch_size))
        get_spectrogram = lambda i: preds.frames[: batch.spectrogram.lengths[:, i], i]
        for _ in range(num_batches):
            indicies = torch.multinomial(weights, self.batch_size, replacement=True).tolist()
            slices = [self._slice(get_spectrogram(i), batch.audio[i]) for i in indicies]
            decoded = [(batch.spans[i].speaker, batch.spans[i].session) for i in indicies]
            speaker = [self.input_encoder.speaker_encoder.encode(d[0]).view(1) for d in decoded]
            session = [self.input_encoder.session_encoder.encode(d).view(1) for d in decoded]
            yield Batch(
                batch=SpectrogramModelBatch(
                    **lib.utils.dataclass_as_dict(batch),
                    predicted_spectrogram=SequenceBatch(preds.frames, batch.spectrogram.lengths),
                ),
                indicies=indicies,
                spectrogram=self._stack([s.spectrogram for s in slices]),
                spectrogram_mask=self._stack([s.spectrogram_mask for s in slices]),
                target_signal=self._stack([s.target_signal for s in slices]),
                signal_mask=self._stack([s.signal_mask for s in slices]),
                speaker=torch.cat(speaker),
                session=torch.cat(session),
            )

    def __iter__(self) -> typing.Iterator[Batch]:
        for spans in self.iterator:
            yield from self._make_batches(spans)

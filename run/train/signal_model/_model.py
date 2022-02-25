import typing

import torch
from hparams import HParam, configurable

from lib import signal_model
from run.data._loader import Session, Speaker


class SignalModel(signal_model.SignalModel):
    """This is a wrapper over `SignalModel` that normalizes the input."""

    @configurable
    def __init__(self, max_speakers: int = HParam(), max_sessions: int = HParam(), *args, **kwargs):
        super().__init__((max_speakers, max_sessions), *args, **kwargs)

    @property
    def speaker_vocab(self) -> typing.Dict[Speaker, int]:
        return typing.cast(typing.Dict[Speaker, int], self.encoder.embed_metadata[0].vocab)

    @property
    def session_vocab(self) -> typing.Dict[typing.Tuple[Speaker, Session], int]:
        vocab = self.encoder.embed_metadata[1].vocab
        return typing.cast(typing.Dict[typing.Tuple[Speaker, Session], int], vocab)

    def __call__(
        self,
        spectrogram: torch.Tensor,
        speaker: typing.List[Speaker],
        session: typing.List[typing.Tuple[Speaker, Session]],
        spectrogram_mask: typing.Optional[torch.Tensor] = None,
        pad_input: bool = True,
    ) -> torch.Tensor:
        seq_metadata = list(zip(speaker, session))
        return super().__call__(spectrogram, seq_metadata, spectrogram_mask, pad_input)


class SpectrogramDiscriminator(signal_model.SpectrogramDiscriminator):
    """This is a wrapper over `SpectrogramDiscriminator` that normalizes the input."""

    @configurable
    def __init__(self, *args, max_speakers: int = HParam(), max_sessions: int = HParam(), **kwargs):
        super().__init__(*args, max_seq_meta_values=(max_speakers, max_sessions), **kwargs)

    @property
    def speaker_vocab(self) -> typing.Dict[Speaker, int]:
        return typing.cast(typing.Dict[Speaker, int], self.encoder.embed_metadata[0].vocab)

    @property
    def session_vocab(self) -> typing.Dict[typing.Tuple[Speaker, Session], int]:
        vocab = self.encoder.embed_metadata[1].vocab
        return typing.cast(typing.Dict[typing.Tuple[Speaker, Session], int], vocab)

    def __call__(
        self,
        spectrogram: torch.Tensor,
        db_spectrogram: torch.Tensor,
        db_mel_spectrogram: torch.Tensor,
        speaker: typing.List[Speaker],
        session: typing.List[typing.Tuple[Speaker, Session]],
    ) -> torch.Tensor:
        seq_metadata = list(zip(speaker, session))
        return super().__call__(spectrogram, db_spectrogram, db_mel_spectrogram, seq_metadata)


def generate_waveform(
    model: SignalModel,
    spectrogram: typing.Iterator[torch.Tensor],
    speaker: typing.List[Speaker],
    session: typing.List[typing.Tuple[Speaker, Session]],
    spectrogram_mask: typing.Optional[typing.Iterator[torch.Tensor]] = None,
) -> typing.Iterator[torch.Tensor]:
    seq_metadata = list(zip(speaker, session))
    return signal_model.generate_waveform(model, spectrogram, seq_metadata, spectrogram_mask)

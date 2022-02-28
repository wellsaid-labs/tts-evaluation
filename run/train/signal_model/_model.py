import typing

import torch
from hparams import HParam, configurable

from lib import signal_model
from lib.utils import PaddingAndLazyEmbedding
from run.data._loader import Session, Speaker


class SignalModel(signal_model.SignalModel):
    """This is a wrapper over `SignalModel` that normalizes the input."""

    @configurable
    def __init__(self, max_speakers: int = HParam(), max_sessions: int = HParam(), *args, **kwargs):
        super().__init__((max_speakers, max_sessions), *args, **kwargs)
        self.speaker_embed = typing.cast(PaddingAndLazyEmbedding, self.encoder.embed_metadata[0])
        self.session_embed = typing.cast(PaddingAndLazyEmbedding, self.encoder.embed_metadata[1])

    @property
    def speaker_vocab(self) -> typing.Dict[Speaker, int]:
        return typing.cast(typing.Dict[Speaker, int], self.speaker_embed.vocab)

    @property
    def session_vocab(self) -> typing.Dict[Session, int]:
        return typing.cast(typing.Dict[Session, int], self.session_embed.vocab)

    def update_speaker_vocab(
        self, speakers: typing.List[Speaker], embeddings: typing.Optional[torch.Tensor] = None
    ):
        return self.speaker_embed.update_tokens(speakers, embeddings)

    def update_session_vocab(
        self,
        sessions: typing.List[Session],
        embeddings: typing.Optional[torch.Tensor] = None,
    ):
        return self.session_embed.update_tokens(sessions, embeddings)

    def __call__(
        self,
        spectrogram: torch.Tensor,
        speaker: typing.List[Speaker],
        session: typing.List[Session],
        spectrogram_mask: typing.Optional[torch.Tensor] = None,
        pad_input: bool = True,
    ) -> torch.Tensor:
        # TODO: Since `Session` contains `Speaker`, do we need to pass both? Or can we simply pass
        # just `Session`?
        seq_metadata = list(zip(speaker, session))
        return super().__call__(spectrogram, seq_metadata, spectrogram_mask, pad_input)


class SpectrogramDiscriminator(signal_model.SpectrogramDiscriminator):
    """This is a wrapper over `SpectrogramDiscriminator` that normalizes the input."""

    @configurable
    def __init__(self, *args, max_speakers: int = HParam(), max_sessions: int = HParam(), **kwargs):
        super().__init__(*args, max_seq_meta_values=(max_speakers, max_sessions), **kwargs)
        self.speaker_embed = typing.cast(PaddingAndLazyEmbedding, self.encoder.embed_metadata[0])
        self.session_embed = typing.cast(PaddingAndLazyEmbedding, self.encoder.embed_metadata[1])

    @property
    def speaker_vocab(self) -> typing.Dict[Speaker, int]:
        return typing.cast(typing.Dict[Speaker, int], self.speaker_embed.vocab)

    @property
    def session_vocab(self) -> typing.Dict[Session, int]:
        return typing.cast(typing.Dict[Session, int], self.session_embed.vocab)

    def __call__(
        self,
        spectrogram: torch.Tensor,
        db_spectrogram: torch.Tensor,
        db_mel_spectrogram: torch.Tensor,
        speaker: typing.List[Speaker],
        session: typing.List[Session],
    ) -> torch.Tensor:
        seq_metadata = list(zip(speaker, session))
        return super().__call__(spectrogram, db_spectrogram, db_mel_spectrogram, seq_metadata)


def generate_waveform(
    model: SignalModel,
    spectrogram: typing.Iterator[torch.Tensor],
    speaker: typing.List[Speaker],
    session: typing.List[Session],
    spectrogram_mask: typing.Optional[typing.Iterator[torch.Tensor]] = None,
) -> typing.Iterator[torch.Tensor]:
    seq_metadata = list(zip(speaker, session))
    return signal_model.generate_waveform(model, spectrogram, seq_metadata, spectrogram_mask)

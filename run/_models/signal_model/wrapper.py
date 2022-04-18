import typing

import torch

from lib.utils import NumeralizePadEmbed
from run._models.signal_model.model import SignalModel, SpectrogramDiscriminator, generate_waveform
from run.data._loader import Session, Speaker


class SignalModelWrapper(SignalModel):
    """This is a wrapper over `SignalModel` that normalizes the input."""

    def __init__(self, max_speakers: int, max_sessions: int, *args, **kwargs):
        super().__init__((max_speakers, max_sessions), *args, **kwargs)

    @property
    def speaker_embed(self) -> NumeralizePadEmbed[Speaker]:
        return typing.cast(NumeralizePadEmbed, self.encoder.embed_metadata[0])

    @property
    def session_embed(self) -> NumeralizePadEmbed[Session]:
        return typing.cast(NumeralizePadEmbed, self.encoder.embed_metadata[1])

    def __call__(
        self,
        spectrogram: torch.Tensor,
        session: typing.List[Session],
        spectrogram_mask: typing.Optional[torch.Tensor] = None,
        pad_input: bool = True,
    ) -> torch.Tensor:
        seq_metadata = [(typing.cast(typing.Hashable, sesh[0].label), sesh) for sesh in session]
        return super().__call__(spectrogram, seq_metadata, spectrogram_mask, pad_input)


class SpectrogramDiscriminatorWrapper(SpectrogramDiscriminator):
    """This is a wrapper over `SpectrogramDiscriminator` that normalizes the input."""

    def __init__(self, *args, max_speakers: int, max_sessions: int, **kwargs):
        super().__init__(*args, max_seq_meta_values=(max_speakers, max_sessions), **kwargs)

    @property
    def speaker_embed(self) -> NumeralizePadEmbed[Speaker]:
        return typing.cast(NumeralizePadEmbed[Speaker], self.encoder.embed_metadata[0])

    @property
    def session_embed(self) -> NumeralizePadEmbed[Session]:
        return typing.cast(NumeralizePadEmbed[Session], self.encoder.embed_metadata[1])

    def __call__(
        self,
        spectrogram: torch.Tensor,
        db_spectrogram: torch.Tensor,
        db_mel_spectrogram: torch.Tensor,
        session: typing.List[Session],
    ) -> torch.Tensor:
        seq_metadata = [(typing.cast(typing.Hashable, sesh[0].label), sesh) for sesh in session]
        return super().__call__(spectrogram, db_spectrogram, db_mel_spectrogram, seq_metadata)


def generate_waveform_wrapper(
    model: SignalModelWrapper,
    spectrogram: typing.Iterable[torch.Tensor],
    session: typing.List[Session],
    spectrogram_mask: typing.Optional[typing.Iterable[torch.Tensor]] = None,
) -> typing.Iterator[torch.Tensor]:
    return generate_waveform(model, spectrogram, session, spectrogram_mask=spectrogram_mask)

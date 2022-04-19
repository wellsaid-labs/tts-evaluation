import typing

import torch

from lib.utils import NumeralizePadEmbed
from run._models.signal_model.model import SignalModel, generate_waveform
from run.data._loader import Session


class SignalModelWrapper(SignalModel):
    """This is a wrapper over `SignalModel` that normalizes the input."""

    def __init__(self, max_speakers: int, max_sessions: int, *args, **kwargs):
        super().__init__((max_speakers, max_sessions), *args, **kwargs)

    @property
    def speaker_embed(self) -> NumeralizePadEmbed[str]:
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
    ) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        seq_metadata = [[typing.cast(typing.Hashable, s[0].label) for s in session], session]
        return super().__call__(spectrogram, seq_metadata, spectrogram_mask, pad_input)


def generate_waveform_wrapper(
    model: SignalModelWrapper,
    spectrogram: typing.Iterable[torch.Tensor],
    session: typing.List[Session],
    spectrogram_mask: typing.Optional[typing.Iterable[torch.Tensor]] = None,
) -> typing.Iterator[torch.Tensor]:
    return generate_waveform(model, spectrogram, session, spectrogram_mask=spectrogram_mask)

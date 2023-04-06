import math
import typing

import torch

from lib.distributed import NumeralizePadEmbed
from run._models.spectrogram_model.containers import Preds
from run._models.spectrogram_model.inputs import (
    Casing,
    Context,
    Inputs,
    InputsWrapper,
    Pronunciation,
    preprocess_inputs,
    preprocess_spans,
)
from run._models.spectrogram_model.model import Mode, SpectrogramModel
from run.data._loader import structures as struc

InputsTyping = typing.Union[InputsWrapper, typing.List[struc.Span], Inputs]


class SpectrogramModelWrapper(SpectrogramModel):
    """This is a wrapper over `SpectrogramModel` that normalizes the input."""

    def __init__(
        self,
        max_tokens: int,
        max_speakers: int,
        max_sessions: int,
        max_dialects: int,
        max_styles: int,
        max_languages: int,
        max_token_embed_size: int,
        *args,
        **kwargs,
    ):
        super().__init__(
            *args,
            max_tokens=max_tokens,
            max_seq_meta_values=(
                max_speakers,
                max_sessions,
                max_dialects,
                max_styles,
                max_languages,
            ),
            max_token_meta_values=(len(Casing) * len(Pronunciation), len(Context)),
            max_token_embed_size=max_token_embed_size,
            **kwargs,
        )

    @property
    def token_embed(self) -> NumeralizePadEmbed[str]:
        # NOTE: `torch.nn.Module` has special hooks for attributes which we avoid by setting this
        # as a `@property`, instead.
        return typing.cast(NumeralizePadEmbed[str], self.encoder.embed_token)

    @property
    def speaker_embed(self) -> NumeralizePadEmbed[str]:
        return typing.cast(NumeralizePadEmbed[str], self.encoder.embed_seq_metadata[0])

    @property
    def session_embed(self) -> NumeralizePadEmbed[struc.Session]:
        return typing.cast(NumeralizePadEmbed[struc.Session], self.encoder.embed_seq_metadata[1])

    @property
    def dialect_embed(self) -> NumeralizePadEmbed[struc.Dialect]:
        return typing.cast(NumeralizePadEmbed[struc.Dialect], self.encoder.embed_seq_metadata[2])

    @property
    def style_embed(self) -> NumeralizePadEmbed[struc.Style]:
        return typing.cast(NumeralizePadEmbed[struc.Style], self.encoder.embed_seq_metadata[3])

    @property
    def language_embed(self) -> NumeralizePadEmbed[struc.Language]:
        return typing.cast(NumeralizePadEmbed[struc.Language], self.encoder.embed_seq_metadata[4])

    @typing.overload
    def __call__(
        self,
        inputs: InputsTyping,
        target_frames: torch.Tensor,
        target_mask: typing.Optional[torch.Tensor] = None,
        mode: typing.Literal[Mode.FORWARD] = Mode.FORWARD,
    ) -> Preds:
        ...  # pragma: no cover

    @typing.overload
    def __call__(
        self,
        inputs: InputsTyping,
        use_tqdm: bool = False,
        token_skip_warning: float = math.inf,
        mode: typing.Literal[Mode.INFER] = Mode.INFER,
    ) -> Preds:
        ...  # pragma: no cover

    @typing.overload
    def __call__(
        self,
        inputs: InputsTyping,
        split_size: float = 32,
        use_tqdm: bool = False,
        token_skip_warning: float = math.inf,
        mode: typing.Literal[Mode.GENERATE] = Mode.GENERATE,
    ) -> typing.Generator[Preds, None, None]:
        ...  # pragma: no cover

    def __call__(
        self,
        inputs: InputsTyping,
        *args,
        mode: typing.Literal[Mode.FORWARD] = Mode.FORWARD,
        **kwargs,
    ) -> typing.Union[typing.Generator[Preds, None, None], Preds]:
        if isinstance(inputs, InputsWrapper):
            inputs = preprocess_inputs(inputs, device=self.encoder.embed_token.weight.device)
        elif isinstance(inputs, list):
            inputs = preprocess_spans(inputs, device=self.encoder.embed_token.weight.device)

        return super().__call__(inputs, *args, mode=mode, **kwargs)
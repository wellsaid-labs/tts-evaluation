import math
import typing

import torch
from hparams import HParam, configurable

from lib import spectrogram_model
from lib.spectrogram_model import Generator, Mode, Preds
from run.data._loader import Session, Speaker


class Inputs(typing.NamedTuple):
    """The model inputs."""

    # Batch of speakers
    speaker: typing.List[Speaker]

    # Batch of recording sessions per speaker
    session: typing.List[typing.Tuple[Speaker, Session]]

    # Batch of sequences of tokens
    tokens: typing.List[typing.List[str]]


class SpectrogramModel(spectrogram_model.SpectrogramModel):
    """This is a wrapper over `SpectrogramModel` that normalizes the input."""

    @configurable
    def __init__(
        self,
        max_tokens: int = HParam(),
        max_speakers: int = HParam(),
        max_sessions: int = HParam(),
        *args,
        **kwargs,
    ):
        super().__init__(max_tokens, (max_speakers, max_sessions), *args, **kwargs)

    @property
    def token_vocab(self) -> typing.Dict[str, int]:
        return typing.cast(typing.Dict[str, int], self.encoder.embed_token.vocab)

    @property
    def speaker_vocab(self) -> typing.Dict[Speaker, int]:
        return typing.cast(typing.Dict[Speaker, int], self.encoder.embed_metadata[0].vocab)

    @property
    def session_vocab(self) -> typing.Dict[typing.Tuple[Speaker, Session], int]:
        vocab = self.encoder.embed_metadata[1].vocab
        return typing.cast(typing.Dict[typing.Tuple[Speaker, Session], int], vocab)

    @typing.overload
    def __call__(
        self,
        inputs: Inputs,
        target_frames: torch.Tensor,
        target_mask: typing.Optional[torch.Tensor] = None,
        mode: typing.Literal[Mode.FORWARD] = Mode.FORWARD,
    ) -> Preds:
        ...  # pragma: no cover

    @typing.overload
    def __call__(
        self,
        inputs: Inputs,
        use_tqdm: bool = False,
        token_skip_warning: float = math.inf,
        mode: typing.Literal[Mode.INFER] = Mode.INFER,
    ) -> Preds:
        ...  # pragma: no cover

    @typing.overload
    def __call__(
        self,
        inputs: Inputs,
        split_size: float = 32,
        use_tqdm: bool = False,
        token_skip_warning: float = math.inf,
        mode: typing.Literal[Mode.GENERATE] = Mode.GENERATE,
    ) -> Generator:
        ...  # pragma: no cover

    def __call__(self, *args, mode: Mode = Mode.FORWARD, **kwargs):
        return super().__call__(*args, mode=mode, **kwargs)

    def forward(self, inputs: Inputs, *args, mode: Mode = Mode.FORWARD, **kwargs):
        seq_metadata = list(zip(inputs.speaker, inputs.session))
        inputs_ = spectrogram_model.Inputs(tokens=inputs.tokens, seq_metadata=seq_metadata)
        return super().forward(inputs_, *args, mode=mode, **kwargs)

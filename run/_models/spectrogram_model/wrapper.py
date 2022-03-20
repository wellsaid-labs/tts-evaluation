import math
import typing

import torch

from lib.utils import NumeralizePadEmbed
from run._models.spectrogram_model.containers import Inputs, Preds
from run._models.spectrogram_model.model import Generator, Mode, SpectrogramModel
from run.data._loader import Session, Speaker


class InputsWrapper(typing.NamedTuple):
    """The model inputs."""

    # Batch of speakers
    speaker: typing.List[Speaker]

    # Batch of recording sessions per speaker
    session: typing.List[Session]

    # Batch of sequences of tokens
    tokens: typing.List[typing.List[str]]


class SpectrogramModelWrapper(SpectrogramModel):
    """This is a wrapper over `SpectrogramModel` that normalizes the input."""

    def __init__(
        self,
        max_tokens: int,
        max_speakers: int,
        max_sessions: int,
        *args,
        **kwargs,
    ):
        super().__init__(max_tokens, (max_speakers, max_sessions), *args, **kwargs)

    @property
    def token_embed(self) -> NumeralizePadEmbed:
        # NOTE: `torch.nn.Module` has special hooks for attributes which we avoid by setting this
        # as a property, instead.
        return self.encoder.embed_token

    @property
    def speaker_embed(self) -> NumeralizePadEmbed:
        return typing.cast(NumeralizePadEmbed, self.encoder.embed_metadata[0])

    @property
    def session_embed(self) -> NumeralizePadEmbed:
        return typing.cast(NumeralizePadEmbed, self.encoder.embed_metadata[1])

    @property
    def token_vocab(self) -> typing.Dict[str, int]:
        return typing.cast(typing.Dict[str, int], self.token_embed.vocab)

    @property
    def speaker_vocab(self) -> typing.Dict[Speaker, int]:
        return typing.cast(typing.Dict[Speaker, int], self.speaker_embed.vocab)

    @property
    def session_vocab(self) -> typing.Dict[Session, int]:
        return typing.cast(typing.Dict[Session, int], self.session_embed.vocab)

    def update_token_vocab(
        self, tokens: typing.List[str], embeddings: typing.Optional[torch.Tensor] = None
    ):
        self.token_embed.update_tokens(tokens, embeddings)

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

    @typing.overload
    def __call__(
        self,
        inputs: InputsWrapper,
        target_frames: torch.Tensor,
        target_mask: typing.Optional[torch.Tensor] = None,
        mode: typing.Literal[Mode.FORWARD] = Mode.FORWARD,
    ) -> Preds:
        ...  # pragma: no cover

    @typing.overload
    def __call__(
        self,
        inputs: InputsWrapper,
        use_tqdm: bool = False,
        token_skip_warning: float = math.inf,
        mode: typing.Literal[Mode.INFER] = Mode.INFER,
    ) -> Preds:
        ...  # pragma: no cover

    @typing.overload
    def __call__(
        self,
        inputs: InputsWrapper,
        split_size: float = 32,
        use_tqdm: bool = False,
        token_skip_warning: float = math.inf,
        mode: typing.Literal[Mode.GENERATE] = Mode.GENERATE,
    ) -> Generator:
        ...  # pragma: no cover

    def __call__(
        self,
        inputs: InputsWrapper,
        *args,
        mode: typing.Literal[Mode.FORWARD] = Mode.FORWARD,
        **kwargs,
    ) -> typing.Union[Generator, Preds]:
        seq_metadata = list(zip(inputs.speaker, inputs.session))
        inputs_ = Inputs(tokens=inputs.tokens, seq_metadata=seq_metadata)
        return super().__call__(inputs_, *args, mode=mode, **kwargs)

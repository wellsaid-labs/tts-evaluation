import enum
import math
import typing

import spacy
import spacy.tokens
import torch
from hparams import HParam, configurable

from lib import spectrogram_model
from lib.spectrogram_model import Generator, Mode, Preds
from lib.utils import PaddingAndLazyEmbedding
from run.data._loader import Session, Speaker


class Inputs(typing.NamedTuple):
    """The model inputs."""

    # Batch of speakers
    speaker: typing.List[Speaker]

    # Batch of recording sessions per speaker
    session: typing.List[Session]

    # Batch of sequences of `Span` which include `Doc` context
    spans: typing.List[typing.Union[spacy.tokens.span.Span, spacy.tokens.doc.Doc]]


class _Casing(enum.Enum):

    LOWER: typing.Final = "lower"
    UPPER: typing.Final = "upper"
    NO_CASING: typing.Final = "no casing"


def _get_case(c: str) -> _Casing:
    assert len(c) == 1
    if c.isupper():
        return _Casing.UPPER
    return _Casing.LOWER if c.islower() else _Casing.NO_CASING


def _preprocess_inputs(inputs: Inputs, num_context_words: int) -> spectrogram_model.Inputs:
    """Preprocess inputs to inputs by including casing, context, and embeddings."""
    token_embeddings: typing.List[torch.Tensor] = []
    token_metadata: typing.List[typing.List[typing.Tuple[_Casing]]] = []
    slices: typing.List[slice] = []
    tokens: typing.List[typing.List[str]] = []
    for span in inputs.spans:
        span = span[:] if isinstance(span, spacy.tokens.doc.Doc) else span
        end = min(span.end + num_context_words, len(span.doc))
        contextual = span.doc[max(span.start - num_context_words, 0) : end]
        start_char = span.start_char - contextual.start_char
        slices.append(slice(start_char, start_char + len(str(span))))
        token_metadata.append([(_get_case(c),) for c in str(contextual)])
        tokens.append(list(str(contextual).lower()))

        # NOTE: Tack on word embeddings for each token
        # TODO: Instead of using `zeros`, what if we tried training a vector, instead?
        embeddings = torch.zeros(len(str(contextual)), span.doc.vector.size)
        for word in contextual:
            word_embedding = torch.from_numpy(word.vector).unsqueeze(0).repeat(len(word), 1)
            embeddings[word.idx : word.idx + len(word)] = word_embedding
        token_embeddings.append(embeddings)

    return spectrogram_model.Inputs(
        tokens=tokens,
        seq_metadata=list(zip(inputs.speaker, inputs.session)),
        token_metadata=token_metadata,
        token_embeddings=token_embeddings,
        slices=slices,
    )


class SpectrogramModel(spectrogram_model.SpectrogramModel):
    """This is a wrapper over `SpectrogramModel` that normalizes the input."""

    @configurable
    def __init__(
        self,
        max_tokens: int = HParam(),
        max_speakers: int = HParam(),
        max_sessions: int = HParam(),
        max_token_embed_size: int = HParam(),
        num_context_words: int = HParam(),
        *args,
        **kwargs,
    ):
        super().__init__(
            *args,
            max_tokens=max_tokens,
            max_seq_meta_values=(max_speakers, max_sessions),
            max_token_meta_values=(len(_Casing),),
            max_token_embed_size=max_token_embed_size,
            **kwargs,
        )
        self.num_context_words = num_context_words
        self.token_embed = self.encoder.embed_token
        self.speaker_embed = typing.cast(
            PaddingAndLazyEmbedding, self.encoder.embed_seq_metadata[0]
        )
        self.session_embed = typing.cast(
            PaddingAndLazyEmbedding, self.encoder.embed_seq_metadata[1]
        )

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

    def __call__(self, inputs: Inputs, *args, mode: Mode = Mode.FORWARD, **kwargs):
        inputs_ = _preprocess_inputs(inputs, self.num_context_words)
        return super().__call__(inputs_, *args, mode=mode, **kwargs)

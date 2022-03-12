import enum
import math
import typing

import spacy
import spacy.tokens
import torch
from hparams import HParam, configurable

from lib import spectrogram_model
from lib.spectrogram_model import Generator, Mode, Preds
from lib.utils import NumeralizePadEmbed
from run.data._loader import Session, Span, Speaker


class _Casing(enum.Enum):

    LOWER: typing.Final = "lower"
    UPPER: typing.Final = "upper"
    NO_CASING: typing.Final = "no casing"


def _get_case(c: str) -> _Casing:
    assert len(c) == 1
    if c.isupper():
        return _Casing.UPPER
    return _Casing.LOWER if c.islower() else _Casing.NO_CASING


def _make_token_embeddings(
    span: typing.Union[spacy.tokens.span.Span, spacy.tokens.doc.Doc], device: torch.device
) -> torch.Tensor:
    """Make a `Tensor` that is an embedding of the `span`."""
    # TODO: Instead of using `zeros`, what if we tried training a vector, instead?
    span = span[:] if isinstance(span, spacy.tokens.doc.Doc) else span
    embeddings = torch.zeros(len(str(span)), span.doc.vector.size, device=device)
    for word in span:
        word_embedding = torch.tensor(word.vector, device=device).unsqueeze(0).repeat(len(word), 1)
        idx = word.idx - span.start_char
        embeddings[idx : idx + len(word)] = word_embedding
    return embeddings


def _append_tokens_and_metadata(
    inputs: spectrogram_model.Inputs,
    span: typing.Union[spacy.tokens.span.Span, spacy.tokens.doc.Doc],
):
    """Preprocess and append `tokens` and `token_metadata` to `inputs`."""
    inputs.token_metadata.append([(_get_case(c),) for c in str(span)])
    inputs.tokens.append(list(str(span).lower()))


def preprocess_spans(
    spans: typing.List[Span], device: torch.device = torch.device("cpu")
) -> spectrogram_model.Inputs:
    """Preprocess inputs to inputs by including casing, context, and embeddings."""
    return_ = spectrogram_model.Inputs([], [], [], [], [])
    for span in spans:
        context = span.spacy_with_context()
        start_char = span.spacy.start_char - context.start_char
        return_.slices.append(slice(start_char, start_char + len(str(span.spacy))))
        return_.seq_metadata.append((span.speaker, span.session))
        _append_tokens_and_metadata(return_, context)
        typing.cast(list, return_.token_embeddings).append(_make_token_embeddings(context, device))
    token_embeddings = torch.nn.utils.rnn.pad_sequence(return_.token_embeddings, batch_first=True)
    return return_._replace(token_embeddings=token_embeddings)


class Inputs(typing.NamedTuple):
    """The model inputs."""

    # Batch of recording sessions per speaker
    session: typing.List[Session]

    # Batch of sequences of `Span` which include `Doc` context
    doc: typing.List[spacy.tokens.doc.Doc]


def preprocess_inputs(
    inputs: Inputs, device: torch.device = torch.device("cpu")
) -> spectrogram_model.Inputs:
    """Preprocess inputs to inputs by including casing, context, and embeddings."""
    return_ = spectrogram_model.Inputs([], [], [], [], [])
    for doc, sesh in zip(inputs.doc, inputs.session):
        return_.slices.append(slice(0, len(str(doc))))
        return_.seq_metadata.append((sesh[0], sesh))
        _append_tokens_and_metadata(return_, doc)
        typing.cast(list, return_.token_embeddings).append(_make_token_embeddings(doc, device))
    token_embeddings = torch.nn.utils.rnn.pad_sequence(return_.token_embeddings, batch_first=True)
    return return_._replace(token_embeddings=token_embeddings)


InputsTyping = typing.Union[Inputs, typing.List[Span], spectrogram_model.Inputs]


class SpectrogramModel(spectrogram_model.SpectrogramModel):
    """This is a wrapper over `SpectrogramModel` that normalizes the input."""

    @configurable
    def __init__(
        self,
        max_tokens: int = HParam(),
        max_speakers: int = HParam(),
        max_sessions: int = HParam(),
        max_token_embed_size: int = HParam(),
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

    @property
    def token_embed(self) -> NumeralizePadEmbed:
        # NOTE: `torch.nn.Module` has special hooks for attributes which we avoid by setting this
        # as a property, instead.
        return self.encoder.embed_token

    @property
    def speaker_embed(self) -> NumeralizePadEmbed:
        return typing.cast(NumeralizePadEmbed, self.encoder.embed_seq_metadata[0])

    @property
    def session_embed(self) -> NumeralizePadEmbed:
        return typing.cast(NumeralizePadEmbed, self.encoder.embed_seq_metadata[1])

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
    ) -> Generator:
        ...  # pragma: no cover

    def __call__(self, inputs: InputsTyping, *args, mode: Mode = Mode.FORWARD, **kwargs):
        if isinstance(inputs, Inputs):
            inputs = preprocess_inputs(inputs, self.encoder.embed_token.weight.device)
        elif isinstance(inputs, list):
            inputs = preprocess_spans(inputs, self.encoder.embed_token.weight.device)

        return super().__call__(inputs, *args, mode=mode, **kwargs)

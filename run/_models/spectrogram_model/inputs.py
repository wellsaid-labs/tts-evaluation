import dataclasses
import enum
import typing

import config as cf
import spacy
import spacy.tokens
import torch

from lib.utils import lengths_to_mask
from run.data._loader import structures as struc


class Casing(enum.Enum):

    LOWER: typing.Final = "lower"
    UPPER: typing.Final = "upper"
    NO_CASING: typing.Final = "no casing"


class Context(enum.Enum):
    """Knowing that the model has to use context words differently from the script, we use this
    to deliminate context words from the voice-over script."""

    CONTEXT: typing.Final = "context"
    SCRIPT: typing.Final = "script"


def _get_case(c: str) -> Casing:
    assert len(c) == 1
    if c.isupper():
        return Casing.UPPER
    return Casing.LOWER if c.islower() else Casing.NO_CASING


@dataclasses.dataclass(frozen=True)
class Inputs:
    """The model inputs.

    TODO: Use `tuple`s so these values cannot be reassigned.
    """

    # Batch of sequences of tokens
    tokens: typing.List[typing.List[typing.Hashable]]

    # Metadata associated with each sequence
    seq_metadata: typing.List[typing.List[typing.Hashable]]

    # Metadata associated with each token in each sequence
    token_metadata: typing.List[typing.List[typing.List[typing.Hashable]]]

    # Embeddings associated with each token in each sequence
    # torch.FloatTensor [batch_size, num_tokens, *]
    token_embeddings: typing.Union[torch.Tensor, typing.List[torch.Tensor]]

    # Slice of tokens in each sequence to be voiced
    slices: typing.List[slice]

    device: torch.device = torch.device("cpu")

    # Number of tokens after `slices` is applied
    # torch.LongTensor [batch_size]
    num_tokens: torch.Tensor = dataclasses.field(init=False)

    # Tokens mask after `slices` is applied
    # torch.BoolTensor [batch_size, num_tokens]
    tokens_mask: torch.Tensor = dataclasses.field(init=False)

    def __post_init__(self):
        indices = [s.indices(len(t)) for s, t in zip(self.slices, self.tokens)]
        num_tokens = [b - a for a, b, _ in indices]
        num_tokens_ = torch.tensor(num_tokens, dtype=torch.long, device=self.device)
        object.__setattr__(self, "num_tokens", num_tokens_)
        object.__setattr__(self, "tokens_mask", lengths_to_mask(num_tokens, device=self.device))

    def reconstruct_text(self, i: int) -> str:
        """Reconstruct text from uncased text, and casing labels."""
        text = typing.cast(typing.List[str], self.tokens[i])
        casing = typing.cast(typing.List[Casing], self.token_metadata[0][i])
        return "".join([t.upper() if c == Casing.UPPER else t for t, c in zip(text, casing)])


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
    inputs: Inputs,
    span: typing.Union[spacy.tokens.span.Span, spacy.tokens.doc.Doc],
    start_char: int,
    end_char: int,
):
    """Preprocess and append `tokens` and `token_metadata` to `inputs`."""
    text = str(span)
    inputs.tokens.append(list(text.lower()))
    inputs.slices.append(slice(start_char, end_char))
    inputs.token_metadata[0].append([_get_case(c) for c in text])
    inputs.token_metadata[1].append([Context.CONTEXT for _ in text])
    for i in range(*inputs.slices[-1].indices(len(text))):
        inputs.token_metadata[1][-1][i] = Context.SCRIPT


def _append_seq_metadata(seq_metadata: typing.List, session: struc.Session):
    """Add metadata about the sequence to the model.

    TODO: Create a named tuple to organize each of these?
    """
    seq_metadata[0].append(session[0].label)
    seq_metadata[1].append(session)
    seq_metadata[2].append(session[0].dialect)
    seq_metadata[3].append(session[0].style)
    seq_metadata[4].append(session[0].language)


def _make_inputs(device: torch.device):
    return Inputs([], [[], [], [], [], []], [[], []], [], [], device)


def preprocess_spans(
    spans: typing.List[struc.Span], device: torch.device = torch.device("cpu")
) -> Inputs:
    """Preprocess inputs to inputs by including casing, context, and embeddings."""
    return_ = _make_inputs(device)
    for span in spans:
        context = span.spacy_with_context(**cf.get())
        start_char = span.spacy.start_char - context.start_char
        _append_seq_metadata(return_.seq_metadata, span.session)
        _append_tokens_and_metadata(return_, context, start_char, start_char + len(str(span.spacy)))
        typing.cast(list, return_.token_embeddings).append(_make_token_embeddings(context, device))
    token_embeddings = torch.nn.utils.rnn.pad_sequence(return_.token_embeddings, batch_first=True)
    return dataclasses.replace(return_, token_embeddings=token_embeddings)


class InputsWrapper(typing.NamedTuple):
    """The model inputs."""

    # Batch of recording sessions per speaker
    session: typing.List[struc.Session]

    # Batch of sequences of `Span` which include `Doc` context
    doc: typing.List[spacy.tokens.doc.Doc]


def preprocess_inputs(inputs: InputsWrapper, device: torch.device = torch.device("cpu")) -> Inputs:
    """Preprocess inputs to inputs by including casing, context, and embeddings."""
    return_ = _make_inputs(device)
    for doc, sesh in zip(inputs.doc, inputs.session):
        _append_seq_metadata(return_.seq_metadata, sesh)
        _append_tokens_and_metadata(return_, doc, 0, len(str(doc)))
        typing.cast(list, return_.token_embeddings).append(_make_token_embeddings(doc, device))
    token_embeddings = torch.nn.utils.rnn.pad_sequence(return_.token_embeddings, batch_first=True)
    return dataclasses.replace(return_, token_embeddings=token_embeddings)

import dataclasses
import enum
import random
import string
import typing

import config as cf
import numpy as np
import spacy
import spacy.tokens
import torch

from lib.text import get_respelling, load_cmudict_syl
from lib.utils import lengths_to_mask
from run.data._loader import structures as struc


class Pronunciation(enum.Enum):

    NORMAL: typing.Final = "normal"
    RESPELLING: typing.Final = "respelling"


class Casing(enum.Enum):

    LOWER: typing.Final = "lower"
    UPPER: typing.Final = "upper"
    NO_CASING: typing.Final = "no casing"


class Context(enum.Enum):
    """Knowing that the model has to use context words differently from the script, we use this
    to deliminate context words from the voice-over script."""

    CONTEXT: typing.Final = "context"
    SCRIPT: typing.Final = "script"


@dataclasses.dataclass(frozen=True)
class Token:
    token: spacy.tokens.token.Token
    is_context: bool
    is_whitespace: bool
    try_to_respell: bool
    # TODO: Should these be added to configuration?
    # NOTE: These prefixes and suffixes are choosen based on spaCy's tokenizer. This ensures that
    # |\\mother\\in\\law\\|" is treated as one token instead of being tokenized.
    prefix: str = dataclasses.field(default="|\\", init=False, repr=False)
    suffix: str = dataclasses.field(default="\\|", init=False, repr=False)
    delim: str = dataclasses.field(default="\\", init=False, repr=False)
    text: str = dataclasses.field(init=False, repr=False)
    pronun: Pronunciation = dataclasses.field(init=False, repr=False)

    def __post_init__(self):
        if not self._is_respelled():
            assert self.prefix not in self.token.text, "`prefix` must be unique"
            assert self.suffix not in self.token.text, "`suffix` must be unique"
        pronun, text = self._get_text()
        object.__setattr__(self, "text", text)
        object.__setattr__(self, "pronun", pronun)

    def _is_respelled(self):
        """Is the `self.token.text` already respelled?"""
        text = self.token.text
        is_respelled = text.startswith(self.prefix) and text.endswith(self.suffix)
        text = text[len(self.prefix) : -len(self.suffix)]
        if is_respelled:
            valid_letters = string.ascii_lowercase + "ə"
            message = "Invalid respelling."
            assert len(text) > 0, message
            assert text[0].lower() in valid_letters, message
            assert text[-1].lower() in valid_letters, message
            assert len(set(text.lower()) - set(list(valid_letters + self.delim))) == 0, message
        return is_respelled

    def _try_respelling(self) -> typing.Optional[str]:
        """Get a respelling of `self.token.text` if possible."""
        respell = get_respelling(self.token.text, load_cmudict_syl(), self.delim)
        return None if respell is None else respell

    def _get_text(self) -> typing.Tuple[Pronunciation, str]:
        """Get the text that represents `token` which maybe respelled."""
        if self.is_whitespace:
            return (Pronunciation.NORMAL, self.token.whitespace_)
        if self._is_respelled():
            return (Pronunciation.RESPELLING, self.token.text[len(self.prefix) : -len(self.suffix)])
        if self.try_to_respell:
            repselling = self._try_respelling()
            if repselling is not None:
                return (Pronunciation.RESPELLING, repselling)
        return (Pronunciation.NORMAL, self.token.text)

    @property
    def embed(self):
        """Get an embedding for this token."""
        assert self.token.tensor is not None
        # TODO: At the moment, if a pronunciation exists, then we don't have data on the original
        # word and it's meaning. In the future, when the original word is not lost
        # we can preserve the original word vector.
        if self.pronun is Pronunciation.RESPELLING or self.is_whitespace:
            return np.zeros(self.token.tensor.shape[0] + self.token.vector.shape[0])
        return np.concatenate((self.token.vector, self.token.tensor))

    @staticmethod
    def _get_case(c: str) -> Casing:
        assert len(c) == 1
        if c.isupper():
            return Casing.UPPER
        return Casing.LOWER if c.islower() else Casing.NO_CASING

    @property
    def casing(self):
        return [(self.pronun, Token._get_case(c)) for c in self.text]

    def __len__(self):
        return len(self.text)


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
        casing = typing.cast(
            typing.List[typing.Tuple[Pronunciation, Casing]], self.token_metadata[0][i]
        )
        return "".join([t.upper() if c == Casing.UPPER else t for t, (_, c) in zip(text, casing)])


SpanDoc = typing.Union[spacy.tokens.span.Span, spacy.tokens.doc.Doc]


def _token_to_tokens(
    token: spacy.tokens.token.Token, span: SpanDoc, respell_prob: float
) -> typing.Tuple[Token, Token]:
    """Convert `token` into `Token`s."""
    is_context = token not in span
    is_whitespace_context = True if token == span[-1] else is_context
    try_to_respell = not is_context and random.random() < respell_prob

    last_token = None if token.i == 0 else token.doc[token.i - 1]
    if last_token is not None and len(last_token.whitespace_) == 0 and not last_token.is_punct:
        try_to_respell = False

    next_token = None if len(token.doc) - 1 == token.i else token.doc[token.i + 1]
    if next_token is not None and len(token.whitespace_) == 0 and not next_token.is_punct:
        try_to_respell = False

    return (
        Token(token, is_context, False, try_to_respell),
        Token(token, is_whitespace_context, True, False),
    )


def _preprocess(
    batch: typing.List[typing.Tuple[struc.Session, SpanDoc, SpanDoc]],
    device: torch.device = torch.device("cpu"),
    respell_prob: float = 0.0,
) -> Inputs:
    """Preprocess `batch` into model `Inputs`.

    NOTE: Preprocess as much as possible here so the model is as fast as possible.

    TODO: Instead of using `zero` embeddings, what if we tried training a vector, instead?
    """
    inputs = Inputs([], [[], [], [], [], []], [[], []], [], [], device)
    for sesh, context_span, span in batch:
        tokens = [t for token in context_span for t in _token_to_tokens(token, span, respell_prob)]

        seq_metadata = (sesh[0].label, sesh, sesh[0].dialect, sesh[0].style, sesh[0].language)
        for i, data in enumerate(seq_metadata):
            inputs.seq_metadata[i].append(data)

        if len(tokens) > 0:
            start_index = next(i for i, t in enumerate(tokens) if not t.is_context)
            start_char = sum(len(t.text) for t in tokens[:start_index])
            end_char = start_char + sum(len(t) for t in tokens if not t.is_context)
        else:
            start_char, end_char = 0, 0
        inputs.slices.append(slice(start_char, end_char))

        chars = [c for t in tokens for c in t.text]
        inputs.tokens.append([c.lower() for c in chars])
        inputs.token_metadata[0].append([c for t in tokens for c in t.casing])
        inputs.token_metadata[1].append([Context.CONTEXT for _ in chars])
        for i in range(*inputs.slices[-1].indices(len(chars))):
            inputs.token_metadata[1][-1][i] = Context.SCRIPT

        if len(tokens) == 0:
            embed = torch.zeros(0, 0, device=device)
        else:
            embed = [torch.tensor(t.embed, device=device, dtype=torch.float32) for t in tokens]
            embed = [e.unsqueeze(0).repeat(len(t), 1) for e, t in zip(embed, tokens)]
            embed = torch.cat(embed)
        typing.cast(list, inputs.token_embeddings).append(embed)

    token_embeddings = torch.nn.utils.rnn.pad_sequence(inputs.token_embeddings, batch_first=True)
    return dataclasses.replace(inputs, token_embeddings=token_embeddings)


def preprocess_spans(spans: typing.List[struc.Span], **kw) -> Inputs:
    return _preprocess([(s.session, cf.partial(s.spacy_context)(), s.spacy) for s in spans], **kw)


class InputsWrapper(typing.NamedTuple):
    """The model inputs."""

    # Batch of recording sessions per speaker
    session: typing.List[struc.Session]

    # Batch of sequences of `Span` which include `Doc` context
    doc: typing.List[spacy.tokens.doc.Doc]


def preprocess_inputs(inputs: InputsWrapper, **kw) -> Inputs:
    return _preprocess([(s, d, d) for s, d in zip(inputs.session, inputs.doc)], **kw)

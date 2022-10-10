import dataclasses
import enum
import typing

import numpy as np
import spacy
import spacy.tokens
import torch

from lib.utils import lengths_to_mask
from run.data._loader import structures as struc


class Pronun(enum.Enum):

    NORMAL: typing.Final = "normal"
    RESPELLING: typing.Final = "respelling"


class Casing(enum.Enum):

    LOWER: typing.Final = "lower"
    UPPER: typing.Final = "upper"
    NO_CASING: typing.Final = "no casing"


def _get_case(c: str) -> Casing:
    assert len(c) == 1
    if c.isupper():
        return Casing.UPPER
    return Casing.LOWER if c.islower() else Casing.NO_CASING


class Context(enum.Enum):
    """Knowing that the model has to use context words differently from the script, we use this
    to deliminate context words from the voice-over script."""

    CONTEXT: typing.Final = "context"
    SCRIPT: typing.Final = "script"


class RespellingError(ValueError):
    pass


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


SpanDoc = typing.Union[spacy.tokens.span.Span, spacy.tokens.doc.Doc]


InputsWrapperTypeVar = typing.TypeVar("InputsWrapperTypeVar")


@dataclasses.dataclass(frozen=True)
class InputsWrapper:
    """The model inputs."""

    # Batch of recording sessions
    session: typing.List[struc.Session]

    # Batch of sequences
    span: typing.List[SpanDoc]

    context: typing.List[SpanDoc]

    # Batch of annotations per sequence
    loudness: typing.List[typing.List[typing.Tuple[slice, int]]]

    rate: typing.List[typing.List[typing.Tuple[slice, float]]]

    respellings: typing.List[typing.List[typing.Tuple[slice, str]]]

    respell_map: typing.List[typing.Dict[spacy.tokens.token.Token, str]] = dataclasses.field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self):
        """
        TODO: Double check invariants before this is inputted into the model...
            - The respellings need to be correctly formatted
            - The respellings need to line up with tokens in `span`.
            - The annotations are sorted.
            - The annotations have no overlaps, at all.
            - The annotations should only be for the `span` object not the `context` object.
            - The annotations should also line up with tokens.
            -
        """
        pass

    def to_xml(self, session_vocab: typing.Dict[struc.Session, int]) -> str:
        """Generate XML from model inputs.

        TODO: Implement to help stringify `InputsWrapper` during training.
        """
        return ""

    @classmethod
    def from_xml(
        cls: typing.Type[InputsWrapperTypeVar], session_vocab: typing.Dict[struc.Session, int]
    ) -> InputsWrapperTypeVar:
        """Parse XML into compatible model inputs.

        TODO: Instead of a `session_vocab`, we could consider having an interface where users
        can submit their own session objects, even, custom ones. While this might be slightly
        more generalizable, it has a number of challenges. For example, the `Session` objects
        have sensitive information, we'd need to desensitize it first.
        """
        return InputsWrapper()


def embed_annotations(
    length: int,
    anno: typing.List[typing.Tuple[slice, typing.Union[int, float]]],
    idx_offset: int = 0,
    val_offset: float = 0,
    val_compression: float = 1,
) -> torch.Tensor:
    """Given annotations for a sequence of `length`, this returns an embedding.

    NOTE: The mask uses 1, -1, and 0. The non-zero values represent an annotation. We cycle between
          1 and -1 to indicate that the annotation has changed.

    Args:
        length: The length of the annotated sequence.
        anno: A list of annotations.
        idx_offset: Offset the annotation indicies.
        val_offset: Offset the annotation values so that they are easier to model.
        val_compression: Compress the annotation values so that they are easier to model.

    Returns:
        torch.FloatTensor [length, 2]
    """
    vals = torch.zeros(length)
    mask = torch.zeros(length)
    mask_val = 1.0
    for slice_, val in anno:
        slice_ = slice(slice_.start + idx_offset, slice_.stop + idx_offset, slice_.step)
        vals[slice_] = val
        mask[slice_] = mask_val
        mask_val *= -1
    vals = (vals + val_offset) / val_compression
    return torch.stack((vals, mask), dim=1)


def preprocess(
    wrap: InputsWrapper,
    loudness_kwargs: typing.Dict,
    rate_kwargs: typing.Dict,
    device: torch.device = torch.device("cpu"),
    loudness_offset: float = 50,
    loudness_compression: float = 50,
    rate_offset: float = 0.1,
    rate_compression: float = 0.1,
) -> Inputs:
    """Preprocess `batch` into model `Inputs`.

    NOTE: This preprocessing layer can be run in a seperate process to prepare data for model
          training.
    NOTE: Contextual word-vectors would likely be more informative than word-vectors; however,
          they are likely not as robust in the presence of OOV words due to intentional
          misspellings. Our users intentionally misspell words to adjust the pronunciation. For that
          reason, using contextual word-vectors is risky.

    TODO: Instead of using `zero` embeddings, what if we tried training a vector, instead?
    TODO: Add offset and compression parameters to config.

    Args:
        batch: A row of data in the batch consists of a Session, the script with context, the
            script without context, and any related annotations expressed as a Tensor.
    """
    inputs = Inputs([], [], [[], []], [], [], device)
    iter_ = zip(wrap.session, wrap.span, wrap.context, wrap.loudness, wrap.rate, wrap.respell_map)
    for sesh, span, context, loudness, rate, respell_map in iter_:
        seq_metadata = [sesh[0].label, sesh, sesh[0].dialect, sesh[0].style, sesh[0].language]
        inputs.seq_metadata.extend([[] for _ in seq_metadata])
        [inputs.seq_metadata[i].append(data) for i, data in enumerate(seq_metadata)]

        start_char = next((t.start_char for t in context if t not in span), 0)
        end_char = (len(span.text) + start_char) - len(context.text)
        inputs.slices.append(slice(start_char, end_char))

        is_respelled = [t in respell_map for t in context]
        tokens = [(respell_map[t] if r else t.text) for t, r in zip(context, is_respelled)]
        chars = [c for t in tokens for c in t]
        casing = [_get_case(c) for c in chars]
        pronun = [
            Pronun.RESPELLING if r else Pronun.NORMAL
            for t, r in zip(tokens, is_respelled)
            for _ in t
        ]
        inputs.tokens.append([c.lower() for c in chars])
        inputs.token_metadata[0].append(list(zip(pronun, casing)))
        inputs.token_metadata[1].append([Context.CONTEXT for _ in chars])
        for i in range(*inputs.slices[-1].indices(len(chars))):
            inputs.token_metadata[1][-1][i] = Context.SCRIPT

        if len(tokens) == 0:
            embed = torch.zeros(0, 0, device=device)
        else:
            embed = [np.concatenate((t.vector, t.tensor)) for t in context]
            embed = [torch.tensor(t, device=device, dtype=torch.float32) for t in embed]
            embed = [e.unsqueeze(0).repeat(len(t), 1) for e, t in zip(embed, tokens)]
            embed = torch.cat(embed)

        loudness_embed = embed_annotations(len(chars), loudness, start_char, *loudness_kwargs)
        rate_embed = embed_annotations(len(chars), rate, start_char, *rate_kwargs)
        # rate_embed (torch.FloatTensor [num_tokens, 2])
        # loudness_embed (torch.FloatTensor [num_tokens, 2])
        # embed (torch.FloatTensor [num_tokens, embedding_size]) →
        # [num_tokens, embedding_size + 4]
        embed = torch.cat((embed, rate_embed, loudness_embed), dim=1)
        typing.cast(list, inputs.token_embeddings).append(embed)

    token_embeddings = torch.nn.utils.rnn.pad_sequence(inputs.token_embeddings, batch_first=True)
    return dataclasses.replace(inputs, token_embeddings=token_embeddings)

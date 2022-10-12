import dataclasses
import enum
import functools
import re
import typing

import config as cf
import numpy as np
import spacy
import spacy.tokens
import torch
from lxml import etree

from lib.utils import lengths_to_mask
from run._config.lang import is_voiced
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
SpanAnnotation = typing.Tuple[slice, float]
SpanAnnotations = typing.List[SpanAnnotation]
TokenAnnotations = typing.Dict[spacy.tokens.token.Token, str]


class AnnotationError(ValueError):
    pass


@functools.lru_cache()
def get_xml_schema():
    xml_schema_doc = etree.parse("schema.xsd", None)
    return etree.XMLSchema(xml_schema_doc)


@dataclasses.dataclass(frozen=True)
class InputsWrapper:
    """The model inputs.

    TODO: This does not allow annotations for context, yet. In the future, depending on the
          purpose of context, it might be helpful to either remove it, or align it with the voiced
          text span. For now, the context is only used during training, to help the model
          with pronunciation. It's not used during inference. This creates a discrepency
          between model usage during inference and training that should be resolved. Either way,
          we should have examples without context, so that the model has some examples that
          directly mimic training. This discrepency could explain why the model struggles more
          with phrases than it does full-sentences (i.e. it may not train on many examples of
          phrases without context).
    """

    # Batch of recording sessions
    session: typing.List[struc.Session]

    # Batch of sequences
    span: typing.List[SpanDoc]

    context: typing.List[SpanDoc]

    # Batch of annotations per sequence
    loudness: typing.List[SpanAnnotations]

    tempo: typing.List[SpanAnnotations]

    respellings: typing.List[TokenAnnotations]

    def __post_init__(self):
        cf.partial(self.check_invariants)()

    def check_invariants(
        self,
        min_loudness: float,
        max_loudness: float,
        min_rate: float,
        max_rate: float,
        valid_respelling_chars: str,
        respelling_delim: str,
    ):
        # NOTE: That model recieves public data through this interface, so, we need to have
        # robust verification and clear error messages for API developers.
        # NOTE: `assert` is used for non-public errors, related to using this object.
        # `AnnotationError`s are used for public-facing errors.
        batch_len = len(self.session)
        for items in (self.span, self.context, self.loudness, self.tempo, self.respellings):
            assert len(items) == batch_len
        for token in self.context:
            assert token in self.span
        for batch_span_annotations in (self.loudness, self.tempo):
            for sesh, span_, annotations in zip(self.session, self.span, batch_span_annotations):
                for prev, annotation in zip([None] + annotations, annotations):
                    # NOTE: The only annotations that are acceptable are non-voiced characters
                    # for pauses or spans for speaking tempo.
                    is_pause = not is_voiced(span_.text[annotation[0]], sesh[0].language)
                    is_valid_span = (
                        span_.char_span(annotation[0].start, annotation[0].stop) is not None
                    )
                    if not is_valid_span and not is_pause:
                        raise AnnotationError("The annotations must wrap words fully.")
                    if prev is not None:
                        assert prev[0].stop < annotation[0].start
        if not all(a[1] >= min_loudness and a[1] <= max_loudness for b in self.loudness for a in b):
            message = "The loudness annotations must be between "
            raise AnnotationError(f"{message} {min_loudness} and {max_loudness} db.")
        if not all(a[1] >= min_rate and a[1] <= max_rate for b in self.tempo for a in b):
            message = "The tempo annotations must be between "
            raise AnnotationError(f"{message} {min_rate} and {max_rate} seconds per character.")
        for span_, token_annotations in zip(self.span, self.respellings):
            for token, annotation in token_annotations.items():
                if token is None:
                    raise AnnotationError("Respelling must wrap a word.")
                assert token in span_
                if len(annotation) == 0:
                    raise AnnotationError("Respelling has no text.")
                if (
                    annotation[0].lower() not in valid_respelling_chars
                    or annotation[-1].lower() not in valid_respelling_chars
                ):
                    message = "Respellings must start and end with one of these chars:"
                    raise AnnotationError(f"{message} {valid_respelling_chars}")
                all_chars = set(list(valid_respelling_chars + respelling_delim))
                if len(set(annotation.lower()) - all_chars) != 0:
                    message = "Respellings must have these chars only:"
                    raise AnnotationError(f"{message} {''.join(all_chars)}")
                if not all(
                    len(set(_get_case(c) for c in syllab)) == 1
                    for syllab in annotation.split(respelling_delim)
                ):
                    raise AnnotationError("Respelling must be capitalized correctly.")

    def to_xml(self, session_vocab: typing.Dict[struc.Session, int]) -> str:
        """Generate XML from model inputs.

        TODO: Implement to help stringify `InputsWrapper` during training.
        """
        return ""

    @classmethod
    def from_xml(
        cls: typing.Type[InputsWrapperTypeVar],
        xml: str,
        span: SpanDoc,
        session_vocab: typing.Dict[int, struc.Session],
    ) -> InputsWrapperTypeVar:
        """Parse XML into compatible model inputs.

        TODO: Instead of a `session_vocab`, we could consider having an interface where users
        can submit their own session objects, even, custom ones. While this might be slightly
        more generalizable, it has a number of challenges. For example, the `Session` objects
        have sensitive information, we'd need to desensitize it first.
        TODO: Verify if the XML errors are interpertable.

        Args:
            xml: The original annotated XML.
            span: The spaCy document built on text from that XML.
            session_vocab: A vocabulary mapping avatar IDs in XML to sessions.
        """
        xml_schema = get_xml_schema()
        root = etree.fromstring(xml, None)
        try:
            xml_schema.assertValid(root)
        except etree.DocumentInvalid as xml_errors:
            raise AnnotationError(f"XML is invalid:\n{xml_errors.error_log}")

        parser = etree.XMLPullParser(events=("start", "end"))
        parser.feed(xml)

        annotations = {"loudness": [], "respell": [], "tempo": []}
        session: typing.Optional[struc.Session] = None
        text: str = ""

        for event, elem in parser.read_events():
            if event == "start":
                if elem.tag == "speak":
                    session = session_vocab[elem.get("avatar")]
                elif elem.tag is not None and elem.get("value") is not None:
                    annotations[elem.tag].append(([len(elem.text)], elem.get("value")))
                if elem.text:
                    text += elem.text
            elif event == "end":
                if elem.tag is not None and elem.tag != "speak":
                    annotations[elem.tag][-1][0].append(len(elem.text))
                if elem.tail:
                    text += elem.tail

        assert text == span.text, "The `Span` must have the same text as the XML."
        assert session is not None

        return cls(
            session=[session],
            span=[span],
            context=[span],
            loudness=[[(slice(*tuple(s)), float(v)) for s, v in annotations["loudness"]]],
            tempo=[[(slice(*tuple(s)), float(v)) for s, v in annotations["tempo"]]],
            respellings=[{span.char_span(*tuple(s)): v for s, v in annotations["respell"]}],
        )


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
    NOTE: Usually, for training, it's helpful if the data is within a range of -1 to 1. This
          function provides a `val_offset` and `val_compression` parameter to adjust the annotation
          range as needed.

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
    tempo_kwargs: typing.Dict,
    device: torch.device = torch.device("cpu"),
) -> Inputs:
    """Preprocess `batch` into model `Inputs`.

    NOTE: This preprocessing layer can be run in a seperate process to prepare data for model
          training.
    NOTE: Contextual word-vectors would likely be more informative than word-vectors; however,
          they are likely not as robust in the presence of OOV words due to intentional
          misspellings. Our users intentionally misspell words to adjust the pronunciation. For that
          reason, using contextual word-vectors is risky.

    TODO: Instead of using `zero` embeddings, what if we tried training a vector, instead?

    Args:
        batch: A row of data in the batch consists of a Session, the script with context, the
            script without context, and any related annotations expressed as a Tensor.
    """
    inputs = Inputs([], [], [[], []], [], [], device)
    iter_ = zip(wrap.session, wrap.span, wrap.context, wrap.loudness, wrap.tempo, wrap.respellings)
    for sesh, span, context, loudness, tempo, respell_map in iter_:
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
        rate_embed = embed_annotations(len(chars), tempo, start_char, *tempo_kwargs)
        # rate_embed (torch.FloatTensor [num_tokens, 2])
        # loudness_embed (torch.FloatTensor [num_tokens, 2])
        # embed (torch.FloatTensor [num_tokens, embedding_size]) →
        # [num_tokens, embedding_size + 4]
        embed = torch.cat((embed, rate_embed, loudness_embed), dim=1)
        typing.cast(list, inputs.token_embeddings).append(embed)

    token_embeddings = torch.nn.utils.rnn.pad_sequence(inputs.token_embeddings, batch_first=True)
    return dataclasses.replace(inputs, token_embeddings=token_embeddings)

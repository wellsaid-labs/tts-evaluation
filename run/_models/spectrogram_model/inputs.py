import dataclasses
import enum
import functools
import html
import pathlib
import typing

import config as cf
import numpy as np
import spacy
import spacy.tokens
import torch
from lxml import etree
from torch.nn.utils.rnn import pad_sequence

from lib.text import XMLType, text_to_xml
from lib.utils import lengths_to_mask, offset_slices
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
    SCRIPT_START: typing.Final = "script_start"
    SCRIPT_STOP: typing.Final = "script_stop"


@dataclasses.dataclass(frozen=True)
class Inputs:
    """Preprocessed inputs for the model.

    TODO: Use `tuple`s so these values cannot be reassigned.
    TODO: Explore naming this in a less generic way and so it does not collide with `input`.
    """

    # Batch of sequences of tokens
    tokens: typing.List[typing.List[typing.Hashable]]

    # Metadata associated with each sequence
    seq_meta: typing.List[typing.List[typing.Hashable]]

    # Metadata associated with each token in each sequence
    token_meta: typing.List[typing.List[typing.List[typing.Hashable]]]

    # Statistics and other numerical measurements associated with each sequence
    # torch.FloatTensor [batch_size, *]
    seq_vectors: torch.Tensor

    # A look up for various slices of `token_vectors`.
    token_vector_idx: typing.Dict[str, slice]

    # Statistics and other numerical measurements associated with each token in each sequence
    # torch.FloatTensor [batch_size, num_tokens, *]
    token_vectors: torch.Tensor

    # Slice of tokens in each sequence to be voiced
    slices: typing.List[slice]

    # The maximum audio length to generate for this text in number of frames.
    # NOTE: This must be a positive value greater or equal to one.
    # torch.LongTensor [batch_size]
    max_audio_len: torch.Tensor

    # Number of tokens before `slices` is applied
    # torch.LongTensor [batch_size]
    num_tokens: torch.Tensor = dataclasses.field(init=False)

    # Number of tokens after `slices` is applied
    # torch.LongTensor [batch_size]
    num_sliced_tokens: torch.Tensor = dataclasses.field(init=False)

    # Tokens mask after `slices` is applied
    # torch.FloatTensor [batch_size, num_tokens]
    sliced_tokens_mask: torch.Tensor = dataclasses.field(init=False)

    def __post_init__(self):
        device = self.seq_vectors.device

        indices = [s.indices(len(t)) for s, t in zip(self.slices, self.tokens)]
        num_tokens = [b - a for a, b, _ in indices]
        num_tokens_ = torch.tensor(num_tokens, dtype=torch.long, device=device)
        object.__setattr__(self, "num_sliced_tokens", num_tokens_)
        object.__setattr__(self, "sliced_tokens_mask", lengths_to_mask(num_tokens, device=device))

        num_tokens = [len(seq) for seq in self.tokens]
        num_tokens_ = torch.tensor(num_tokens, dtype=torch.long, device=device)
        object.__setattr__(self, "num_tokens", num_tokens_)

        self.check_invariants()

    def check_invariants(self):
        # NOTE: Ensure all tensors are on the same `device`.
        tensors = (
            self.seq_vectors,
            self.token_vectors,
            self.max_audio_len,
            self.num_sliced_tokens,
            self.sliced_tokens_mask,
        )
        assert len(set(t.device for t in tensors)) == 1

        # NOTE: Double-check typing
        assert self.sliced_tokens_mask.dtype == torch.bool
        assert self.token_vectors.dtype == torch.float32
        assert self.seq_vectors.dtype == torch.float32
        assert self.num_sliced_tokens.dtype == torch.long

        # NOTE: Double-check sizing.
        assert len(self.tokens) == len(self)
        assert len(self.seq_meta) == len(self)
        assert len(self.token_meta) == len(self)
        assert len(self.slices) == len(self)
        if len(self) > 0:
            max_num_tokens = int(self.num_tokens.max())
            max_num_sliced_tokens = int(self.num_sliced_tokens.max())
            token_vector_len = max(v.stop for v in self.token_vector_idx.values())
            assert self.max_audio_len.shape == (len(self),)
            assert self.num_sliced_tokens.shape == (len(self),)
            assert self.sliced_tokens_mask.shape == (len(self), max_num_sliced_tokens)
            assert self.token_vectors.shape == (len(self), max_num_tokens, token_vector_len)
            assert self.seq_vectors.shape[0] == len(self)

        if len(self) > 0:
            # NOTE: Double-check sizing.
            assert all(len(seq) != 0 for seq in self.tokens)
            assert all(len(meta) == self.num_seq_meta for meta in self.seq_meta)
            assert all(len(meta) == self.num_token_meta for meta in self.token_meta)
            assert all(
                len(seq) == len(self.tokens[i])
                for i, meta in enumerate(self.token_meta)
                for seq in meta
            )

            # NOTE: Double check values.
            assert torch.equal(self.sliced_tokens_mask.sum(dim=1), self.num_sliced_tokens)
            assert all(len(s) == self.num_tokens[i] for i, s in enumerate(self.tokens))
            assert all(len(t) >= s.stop for t, s in zip(self.tokens, self.slices))
            assert all(s.start < s.stop and s.step is None for s in self.slices)
            assert all(
                s.stop - s.start == self.num_sliced_tokens[i] for i, s in enumerate(self.slices)
            )

            # NOTE: Double check `token_vector_idx` name space.
            slices = sorted(list(self.token_vector_idx.values()), key=lambda s: s.start)
            assert slices[0].start == 0
            assert slices[-1].stop == self.token_vectors.shape[2]
            assert all(s.start < s.stop and s.step is None for s in slices)
            assert all(a.stop == b.start for a, b in zip(slices, slices[1:]))

    @property
    def num_token_meta(self):
        """Get the number of metadata values per token."""
        return len(self.token_meta[0])

    @property
    def num_seq_meta(self):
        """Get the number of metadata values per sequence."""
        return len(self.seq_meta[0])

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, key: typing.Any):
        if not isinstance(key, (slice, int)):
            raise TypeError
        if isinstance(key, int):
            self.tokens[key]  # NOTE: Raise `IndexError` if needed.
            key = slice(key, key + 1)
        max_num_tokens = self.num_tokens[key].max()
        return Inputs(
            tokens=self.tokens[key],
            seq_meta=self.seq_meta[key],
            token_meta=self.token_meta[key],
            token_vector_idx=self.token_vector_idx,
            seq_vectors=self.seq_vectors[key],
            token_vectors=self.token_vectors[key, :max_num_tokens],
            slices=self.slices[key],
            max_audio_len=self.max_audio_len[key],
        )

    def get_token_vec(self, name, size: typing.Optional[int] = None) -> torch.Tensor:
        """Retrieve a slice of `self.token_vectors` by `name` as set in `self.token_vector_idx`.

        Args:
            ...
            size: An optional `size` to pad to.

        Returns:
            torch.FloatTensor [batch_size, num_tokens, *]
        """
        embed = self.token_vectors[:, :, self.token_vector_idx[name]]
        if size is not None and embed.shape[2] != size:
            message = f"The `{name}` ({embed.shape[2]}) size must be smaller or equal to {size}."
            assert embed.shape[2] <= size, message
            embed_ = torch.zeros(*embed.shape[:2], size, device=embed.device)
            embed_[:, :, : embed.shape[2]] = embed
            embed = embed_
        return embed


SpanDoc = typing.Union[spacy.tokens.span.Span, spacy.tokens.doc.Doc]
Normalize = typing.Callable[[float], float]
SliceAnno = typing.Tuple[slice, float]
SliceAnnos = typing.List[SliceAnno]
NormSliceAnno = typing.List[typing.Tuple[slice, float, float]]
TokenAnnos = typing.Dict[spacy.tokens.token.Token, str]
InputsWrapperTypeVar = typing.TypeVar("InputsWrapperTypeVar")


class PublicValueError(ValueError):
    pass


@functools.lru_cache()
def get_xml_schema():
    xml_schema_doc = etree.parse(pathlib.Path(__file__).parent / "schema.xsd", None)
    return etree.XMLSchema(xml_schema_doc)


class _Schema(enum.Enum):
    # NOTE: Schema tag names
    RESPELL: typing.Final = "respell"
    LOUDNESS: typing.Final = "loudness"
    TEMPO: typing.Final = "tempo"
    SPEAK: typing.Final = "speak"

    # NOTE: Schema attribute names
    _VALUE: typing.Final = "value"

    def __str__(self):
        return str(self.value)


def _idx(span: SpanDoc, token: spacy.tokens.token.Token) -> int:
    """Get the character offset for `token` relative to `span`."""
    start_char = span.start_char if isinstance(span, spacy.tokens.span.Span) else 0
    return token.idx - start_char


@dataclasses.dataclass(frozen=True)
class InputsWrapper:
    """The model inputs before processing.

    This wrapper is an interface for model inputs before they are further processed. It can be
    directly created from XML. It can also be stringified back into XML.

    TODO: This does not allow annotations for context, yet. In the future, depending on the
          purpose of context, it might be helpful to either remove it, or align it with the voiced
          text span. For now, the context is only used during training, to help the model
          with pronunciation. It's not used during inference. This creates a discrepency
          between model usage during inference and training that should be resolved. Either way,
          we should have examples without context, so that the model has some examples that
          directly mimic training. This discrepency could explain why the model struggles more
          with phrases than it does full-sentences (i.e. it may not train on many examples of
          phrases without context).
    TODO: Do we need context? spaCy is already factoring in context that would allow the model
          to predict the POS and other linguistic features. What's the purpose of further having
          context? This also creates a discrepency between training and inference that we need
          to navigate.
    """

    # Batch of recording sessions
    session: typing.List[struc.Session]

    # Batch of sequences
    span: typing.List[SpanDoc]

    context: typing.List[SpanDoc]

    # Batch of annotations per sequence
    loudness: typing.List[SliceAnnos]

    tempo: typing.List[SliceAnnos]

    respells: typing.List[TokenAnnos]

    def __post_init__(self):
        cf.partial(self.check_invariants)()

    def check_invariants(
        self,
        min_loudness: float,
        max_loudness: float,
        min_tempo: float,
        max_tempo: float,
        valid_respelling_chars: str,
        respelling_delim: str,
    ):
        # TODO: This allows for overlapping XML tags to overlap like loudness tags overlapping with
        # tempo tags, this is improper. While this would be easy to check for, the ideal resolution
        # would be to provide a convient function to resolve this during instantiation, so the
        # client does not need to worry about instantiating `InputsWrapper` in a strictly correct
        # way.
        # NOTE: That model recieves public data through this interface, so, we need to have
        # robust verification and clear error messages for API developers.
        # NOTE: `assert` is used for non-public errors, related to using this object.
        # `AnnotationError`s are used for public-facing errors.
        for field in dataclasses.fields(self):
            assert len(self.session) == len(getattr(self, field.name))

        # NOTE: Check that `span` isn't zero.
        for span in self.span:
            if len(span) == 0:
                raise PublicValueError("There must be text.")

        # NOTE: Check that `context` fully contains `span`...
        for context, span in zip(self.context, self.span):
            no_context = isinstance(span, spacy.tokens.doc.Doc)
            has_context = context is not span
            assert has_context or no_context
            assert all(token in context for token in span)
            assert sum(token in span for token in context) == len(span)
            # NOTE: It's possible that a `Doc` object does not pass this check, since some
            # characters are not considered spaCy tokens, like white spaces. This selects all the
            # spaCy tokens and checks if they represent the entire `Doc`.
            assert context.text == context[:].text
            assert span.text == span[:].text

        # NOTE: Check that annotations are sorted and wrap full words.
        for batch_span_annotations in (self.loudness, self.tempo):
            for sesh, span_, annotations in zip(self.session, self.span, batch_span_annotations):
                for prev, annotation in zip([None] + annotations, annotations):
                    # NOTE: The only annotations that are acceptable are non-voiced characters
                    # for pauses or spans for speaking tempo.
                    is_pause = not is_voiced(span_.text[annotation[0]], sesh[0].language)
                    assert annotation[0].start <= annotation[0].stop
                    assert annotation[0].step is None
                    assert annotation[0].start >= 0 and annotation[0].stop <= len(span_.text)
                    indices = annotation[0].indices(len(span_.text))
                    annotation_len = indices[1] - indices[0]
                    if annotation_len == 0:
                        raise PublicValueError("The annotations must wrap text.")
                    # NOTE: For these annotations we accept wrappings that include additional
                    # punctuation on either side of a word; however, we don't accept partially
                    # wrapped tokens.
                    doc = span_.as_doc() if isinstance(span_, spacy.tokens.span.Span) else span_
                    char_span = doc.char_span(indices[0], indices[1], alignment_mode="expand")
                    is_valid_span = char_span is not None and len(char_span.text) <= annotation_len
                    if not is_valid_span and not is_pause:
                        raise PublicValueError("The annotations must wrap words fully.")
                    if prev is not None:
                        assert prev[0].stop <= annotation[0].start, f"{prev}, {annotation}"

        # NOTE: Check that the annotation values are in the right range.
        for name, annotation_batch, min_, max_ in (
            ("Loudness", self.loudness, min_loudness, max_loudness),
            ("Tempo", self.tempo, min_tempo, max_tempo),
        ):
            for annotations in annotation_batch:
                if len(annotations) > 0:
                    min_seen = min(a[1] for a in annotations)
                    max_seen = max(a[1] for a in annotations)
                    message = f"{name} must be between {min_} and {max_}, got: "
                    if min_seen < min_:
                        raise PublicValueError(message + str(min_seen))
                    if max_seen > max_:
                        raise PublicValueError(message + str(max_seen))

        # NOTE: Check that respells are correctly formatted and wrap words entirely.
        for span_, token_annotations in zip(self.span, self.respells):
            for token, annotation in token_annotations.items():
                if token is None:
                    raise PublicValueError("Respelling must wrap a word.")
                assert token in span_
                if len(annotation) == 0:
                    raise PublicValueError("Respelling has no text.")
                if (
                    annotation[0].lower() not in valid_respelling_chars
                    or annotation[-1].lower() not in valid_respelling_chars
                ):
                    message = "Respellings must start and end with one of these chars:"
                    raise PublicValueError(f"{message} {valid_respelling_chars}")
                all_chars = set(list(valid_respelling_chars + respelling_delim))
                if len(set(annotation.lower()) - all_chars) != 0:
                    message = "Respellings must have these chars only:"
                    raise PublicValueError(f"{message} {''.join(all_chars)}")
                if not all(
                    len(set(_get_case(c) for c in syllab)) == 1
                    for syllab in annotation.split(respelling_delim)
                ):
                    raise PublicValueError("Respelling must be capitalized correctly.")

    def __len__(self):
        return len(self.session)

    def to_xml(
        self,
        i: int,
        session_vocab: typing.Optional[typing.Dict[struc.Session, int]] = None,
        include_context: bool = False,
    ) -> XMLType:
        """Generate XML from model inputs.

        NOTE: Due to the possibility of an overlap, `from_xml` and `to_xml` will not nessecarily
              generate the same XML. There is more than one way to achieve the same outcome in XML.

        NOTE: There are options for creating XML with Python programatically; however, they are
              not much better than string concatenation, in this case:
              https://www.geeksforgeeks.org/create-xml-documents-using-python/
              For example, we would still need to use strings to identify the tags.

        Args:
            i: The index of `InputsWrapper` to choose.
            session_vocab: A vocabulary mapping avatar IDs in XML to sessions. If `session_vocab`
                is not provided, -1 is used for the avatar id, as a null value.
            include_context: A convience method for including additional context surrounding
                the XML. Keep in mind, this will invalidate the XML.

        Returns: An XML string.
        """
        span = self.span[i]
        context = self.context[i]
        open_ = lambda tag, value: f"<{tag} {_Schema._VALUE}='{value}'>"
        close = lambda tag: f"</{tag}>"
        respells = self.respells[i].items()
        annotations = [(close(_Schema.RESPELL), _idx(span, t) + len(t)) for t, _ in respells]
        annotations += [(close(_Schema.TEMPO), s.stop) for s, _ in self.tempo[i]]
        annotations += [(close(_Schema.LOUDNESS), s.stop) for s, _ in self.loudness[i]]
        annotations += [(open_(_Schema.LOUDNESS, a), s.start) for s, a in self.loudness[i]]
        annotations += [(open_(_Schema.TEMPO, a), s.start) for s, a in self.tempo[i]]
        annotations += [(open_(_Schema.RESPELL, a), _idx(span, t)) for t, a in respells]
        # TODO: Ensure that this is sorted by specificity, so the most specific tag is opened
        # and closed, first.
        annotations = sorted(annotations, key=lambda k: k[1])
        indices = [0] + [i for _, i in annotations] + [len(span.text)]
        parts = [span.text[i:j] for i, j in zip(indices, indices[1:] + [None])]
        start = open_(_Schema.SPEAK, session_vocab[self.session[i]] if session_vocab else -1)
        stop = close(_Schema.SPEAK)
        annotations = [start] + [a for (a, _) in annotations] + [stop]
        text = "".join("".join((a, text_to_xml(p))) for p, a in zip(parts, annotations))
        if include_context and isinstance(span, spacy.tokens.span.Span):
            start_char = next((_idx(context, t) for t in context if t in span), 0)
            prefix = text_to_xml(context.text[:start_char])
            suffix = text_to_xml(context.text[start_char + len(span.text) :])
            text = f"{prefix}{text}{suffix}"
        return XMLType(text)

    def __getitem__(self, key: typing.Any):
        """Get the ith item in `self`."""
        if not isinstance(key, (slice, int)):
            raise TypeError
        if isinstance(key, int):
            self.session[key]  # NOTE: Raise `IndexError` if needed.
            key = slice(key, key + 1)
        fields = dataclasses.fields(self)
        return self.__class__(**{f.name: getattr(self, f.name)[key] for f in fields})

    @classmethod
    def from_strict_xml(
        cls: typing.Type[InputsWrapperTypeVar],
        xml: XMLType,
        span: SpanDoc,
        session_vocab: typing.Dict[int, struc.Session],
        context: typing.Optional[SpanDoc] = None,
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
        assert span.text == span[:].text, "The text must be stripped."

        # TODO: Make sure that annotations within annotations are not accepted. The model isn't
        # trained with that in mind. We don't allow overlapping annotations, either. It'd be
        # another extreneous level of complexity to explain to customers.
        xml_schema = get_xml_schema()
        root = etree.fromstring(xml, None)
        try:
            xml_schema.assertValid(root)
        except etree.DocumentInvalid as xml_errors:
            raise PublicValueError(f"XML is invalid:\n{xml_errors.error_log}")

        parser = etree.XMLPullParser(events=("start", "end"))
        parser.feed(xml)

        annotations = {_Schema.LOUDNESS: [], _Schema.RESPELL: [], _Schema.TEMPO: []}
        session: typing.Optional[struc.Session] = None
        text: str = ""
        for event, elem in parser.read_events():
            if event == "start":
                if elem.tag == str(_Schema.SPEAK):
                    session = session_vocab[int(elem.get(str(_Schema._VALUE)))]
                elif elem.tag is not None and elem.get(str(_Schema._VALUE)) is not None:
                    annotation = ([len(text)], elem.get(str(_Schema._VALUE)))
                    annotations[_Schema[elem.tag.upper()]].append(annotation)
                if elem.text and len(text) == 0:
                    text += html.unescape(typing.cast(str, elem.text).lstrip())
                elif elem.text:
                    text += html.unescape(elem.text)
            elif event == "end":
                if elem.tag is not None and elem.tag != str(_Schema.SPEAK):
                    annotations[_Schema[elem.tag.upper()]][-1][0].append(len(text))
                if elem.tail:
                    text += html.unescape(elem.tail)

        assert text == span.text, "The `Span` must have the same text as the XML."
        assert session is not None

        respells = {}
        for slice_, value in annotations[_Schema.RESPELL]:
            token = span.char_span(*tuple(slice_))
            if token is None or len(token) != 1:
                raise PublicValueError("Respelling must wrap a single word.")
            respells[token[0]] = value

        slice_annos: typing.Dict[_Schema, SliceAnnos] = {}
        for tag in [_Schema.LOUDNESS, _Schema.TEMPO]:
            if not all(len(s) == 2 for s, _ in annotations[tag]):
                raise PublicValueError(f"The {tag.value} annotations cannot not be nested.")
            try:
                annos = [(s, float(v)) for s, v in annotations[tag]]
            except ValueError:
                raise PublicValueError(f"The {tag.value} annotations must be numerical.")
            slice_annos[tag] = [(slice(*tuple(s)), v) for s, v in annos]

        return cls(
            session=[session],
            span=[span],
            context=[span if context is None else context],
            loudness=[slice_annos[_Schema.LOUDNESS]],
            tempo=[slice_annos[_Schema.TEMPO]],
            respells=[respells],
        )

    @classmethod
    def from_xml(
        cls: typing.Type[InputsWrapperTypeVar],
        xml: XMLType,
        span: SpanDoc,
        session: struc.Session,
        context: typing.Optional[SpanDoc] = None,
    ) -> InputsWrapperTypeVar:
        """Parse XML into compatible model inputs, that may not have a root element."""
        if not xml.startswith(f"<{_Schema.SPEAK}"):
            xml = XMLType(f"<{_Schema.SPEAK} {_Schema._VALUE}='{-1}'>{xml}</{_Schema.SPEAK}>")
        input_ = InputsWrapper.from_strict_xml(xml, span, {-1: session}, context)
        return typing.cast(InputsWrapperTypeVar, input_)

    @classmethod
    def from_xml_batch(
        cls: typing.Type[InputsWrapperTypeVar],
        xml: typing.List[XMLType],
        span: typing.List[SpanDoc],
        session: typing.List[struc.Session],
        context: typing.Optional[typing.List[SpanDoc]] = None,
    ) -> InputsWrapperTypeVar:
        """Parse a batch of XML into compatible model inputs, that may not have a root element."""
        all_ = {f.name: [] for f in dataclasses.fields(InputsWrapper)}
        for args in zip(xml, span, session, span if context is None else context):
            input_ = InputsWrapper.from_xml(*args)
            for key, val in all_.items():
                val.extend(getattr(input_, key))
        return cls(**all_)


def _norm_annos(
    annos: SliceAnnos,
    norm_len: Normalize,
    norm_val: Normalize,
    updates: typing.List[typing.Tuple[slice, int]],
    char_offset: int,
) -> NormSliceAnno:
    """Normalize and adjust annotations.

    Args:
        annos: Annotations mapping slices to values.
        norm_len: A function to normalize the annotation length.
        norm_val: A function to normalize the annotation values.
        updates: Updates to make to the underlying `annos` index, adjusting the slices.
        char_offset: The number of characters to offset `annos`.
    """
    return [
        (
            slice(u.start + char_offset, u.stop + char_offset),
            norm_len(s.stop - s.start),
            norm_val(v),
        )
        for u, (s, v) in zip(offset_slices([s for s, _ in annos], updates), annos)
    ]


def _norm_input(
    inp: InputsWrapper,
    start_char: int,
    norm_anno_len: Normalize,
    norm_anno_loudness: Normalize,
    norm_sesh_loudness: Normalize,
    norm_tempo: Normalize,
):
    """Normalize `inp` values to a standard range for training, usually -1 to 1."""
    # NOTE: Adjust annotation slices after respellings have been added to the text.
    respell_updates = [
        (_char_slice(inp.span[0], token), len(respell))
        for token, respell in inp.respells[0].items()
    ]
    # NOTE: The context is not annotated so we need to offset the annotations by `start_char`.
    args = (respell_updates, start_char)
    return (
        _norm_annos(inp.loudness[0], norm_anno_len, norm_anno_loudness, *args),
        _norm_annos(inp.tempo[0], norm_anno_len, norm_tempo, *args),
        norm_sesh_loudness(inp.session[0].loudness),
        norm_tempo(inp.session[0].tempo),
    )


def _slice_seq(
    slices: typing.List[typing.Tuple[slice, float]], length: int, **kwargs
) -> torch.Tensor:
    """Create a 1-d `Tensor` representing `slices`."""
    sequence = torch.zeros(length, **kwargs)
    for slice_, val in slices:
        sequence[slice_] = val
    return sequence


def _anno_vector(
    anno: NormSliceAnno, avg: float, **kwargs
) -> typing.Tuple[torch.Tensor, torch.Tensor]:
    """Create an sequence with a corresponding annotation vector for each token.

    Args:
        anno: A list of slices with the corresponding length and value.
        avg: A baseline average value for this annotation for the model to reference.
        kwargs: Additional key-word arguments pased to `_slice_seq`

    Returns:
        (torch.FloatTensor [length, num_features])
        (torch.FloatTensor [length, 1]): This is 1 when there is an annotation and 0 otherwise.
    """
    # NOTE: The annotation length annotation helps the model understand how "strict" the annotation
    # is. A short annotation does not have much room to deviate while a long one does.
    anno_vector = (
        _slice_seq([(s, value) for s, _, value in anno], **kwargs),
        _slice_seq([(s, avg) for s, _, _ in anno], **kwargs),
        _slice_seq([(s, length) for s, length, _ in anno], **kwargs),
    )
    anno_vector = torch.stack(anno_vector, dim=1)
    anno_mask = _slice_seq([(slice_, 1) for slice_, _, _ in anno], **kwargs).unsqueeze(1)
    return anno_vector, anno_mask


def _word_vector(span: SpanDoc, tokens: typing.List[str], **kwargs) -> torch.Tensor:
    """Create a sequence with a corresponding word vector for each token.

    Args:
        ...
        tokens (str): A token associated with each token in `span`.

    Returns:
        (torch.FloatTensor [length, num_features])
        (torch.FloatTensor [length, 1]): This is 1 when there is an annotation and 0 otherwise.
    """
    word_vector = []
    for token in span:
        assert token.tensor is not None
        word_vector.append(np.concatenate((token.tensor, token.vector)))  # type: ignore
        word_vector.append(np.zeros(token.vector.shape[0] + token.tensor.shape[0]))
    word_vector = word_vector[:-1]  # NOTE: Discard the trailing whitespace
    word_vector = [torch.tensor(t, **kwargs, dtype=torch.float32) for t in word_vector]
    return torch.cat([e.unsqueeze(0).repeat(len(t), 1) for e, t in zip(word_vector, tokens)])


def _token_vector(
    wrap: InputsWrapper,
    norm_loudness: NormSliceAnno,
    norm_tempo: NormSliceAnno,
    norm_sesh_loudness: float,
    norm_sesh_tempo: float,
    spacy_tokens: typing.List[str],
    **kwargs,
) -> typing.Tuple[typing.Dict[str, slice], torch.Tensor]:
    """Create a vector for each token with various annotations.

    Args:
        ...
        norm_loudness: Normalized loudness annotations.
        norm_tempo: Normalized tempo annotations.
        norm_sesh_loudness: Normalized loudness session average.
        norm_sesh_tempo: Normalized tempo session average.
        spacy_tokens: A list of all the spaCy tokens in the input.

    Returns:
        indicies: A name space to lookup various annotations.
        tensor: The annotation values combined together.
    """
    num_chars = sum(len(t) for t in spacy_tokens)
    anno_kwargs = dict(length=num_chars, **kwargs)
    loudness_vector, loudness_mask = _anno_vector(norm_loudness, norm_sesh_loudness, **anno_kwargs)
    tempo_vector, tempo_mask = _anno_vector(norm_tempo, norm_sesh_tempo, **anno_kwargs)
    word_vector = _word_vector(wrap.context[0], spacy_tokens, **kwargs)
    annos = dict(
        loudness_vector=loudness_vector,  # torch.FloatTensor [total_chars, 3]
        loudness_mask=loudness_mask,  # torch.FloatTensor [total_chars, 1]
        tempo_vector=tempo_vector,  # torch.FloatTensor [total_chars, 3]
        tempo_mask=tempo_mask,  # torch.FloatTensor [total_chars, 1]
        word_vector=word_vector,  # torch.FloatTensor [total_chars, 396]
    )
    indicies, offset = {}, 0
    for name, tensor in annos.items():
        indicies[name] = slice(offset, offset + tensor.shape[1])
        offset += tensor.shape[1]
    return indicies, torch.cat(list(annos.values()), dim=1)


def _char_slice(span: SpanDoc, token: spacy.tokens.token.Token) -> slice:
    """Get a `slice` for the characters represented by `token`."""
    return slice(_idx(span, token), _idx(span, token) + len(token.text))


def preprocess(
    wrap: InputsWrapper,
    get_max_audio_len: typing.Callable[[str], int],
    norm_anno_len: Normalize,
    norm_anno_loudness: Normalize,
    norm_sesh_loudness: Normalize,
    norm_tempo: Normalize,
    **kwargs,
) -> Inputs:
    """Preprocess `batch` into model `Inputs`.

    NOTE: This preprocessing layer can be run in a seperate process to prepare data for model
          training.

    TODO: Instead of using `zero` embeddings, what if we tried training a vector, instead?

    Args:
        wrap: A raw batch of data that needs to be preprocessed.
        get_max_audio_len: A callable for determining the maximum audio length in frames for
            a given piece of text.
        norm_loudness: Normalized loudness annotations.
        norm_tempo: Normalized tempo annotations.
        norm_sesh_loudness: Normalized loudness session average.
        norm_sesh_tempo: Normalized tempo session average.
    """
    empty = torch.empty(0, **kwargs)
    max_audio_len, token_vectors, token_vector_idx, seq_vectors = [], [], {}, []
    result = Inputs([], [], [], empty, {}, empty, [], empty)  # Preprocessed `Inputs`.
    for item in wrap:
        spkr, sesh = item.session[0].spkr, item.session[0]
        result.seq_meta.append([spkr.label, sesh, spkr.dialect, spkr.style, spkr.language])

        spacy_tokens: typing.List[str] = []
        pronun: typing.List[Pronun] = []
        cntxt: typing.List[Context] = []
        respells, span = item.respells[0], item.span[0]
        for tk in item.context[0]:
            spacy_tokens.extend((respells[tk] if tk in respells else tk.text, tk.whitespace_))
            pronun.extend((Pronun.RESPELLING if tk in respells else Pronun.NORMAL, Pronun.NORMAL))
            cntxt.append(Context.SCRIPT if tk in span else Context.CONTEXT)
            cntxt.append(Context.SCRIPT if tk in span and tk != span[-1] else Context.CONTEXT)

        # NOTE: Discard the trailing whitespace
        pronun, spacy_tokens, cntxt = pronun[:-1], spacy_tokens[:-1], cntxt[:-1]

        chars = [c for t in spacy_tokens for c in t]
        casing = [_get_case(c) for c in chars]
        pronun = [p for t, p in zip(spacy_tokens, pronun) for _ in range(len(t))]
        cntxt = [c for t, c in zip(spacy_tokens, cntxt) for _ in range(len(t))]
        start_char = next(i for i, c in enumerate(cntxt) if c is Context.SCRIPT)
        end_char = start_char + cntxt.count(Context.SCRIPT)
        cntxt[start_char] = Context.SCRIPT_START
        cntxt[end_char - 1] = Context.SCRIPT_STOP

        result.slices.append(slice(start_char, end_char))
        # NOTE: This does not consider annotations because it is the maximum audio length found
        # in the dataset, overall. The model shouldn't work well past that.
        max_audio_len.append(get_max_audio_len(span.text))
        result.tokens.append([c.lower() for c in chars])
        # NOTE: `Casing` has a different meaning in a `RESPELLING` versus generally, so it's we
        # merge them together.
        pronun_casing = list(zip(pronun, casing))
        result.token_meta.append([pronun_casing, cntxt])  # type: ignore

        normalized = _norm_input(
            item,
            start_char,
            norm_anno_len,
            norm_anno_loudness,
            norm_sesh_loudness,
            norm_tempo,
        )
        token_vector_idx, token_vector = _token_vector(item, *normalized, spacy_tokens, **kwargs)
        token_vectors.append(token_vector)
        _, _, sesh_loudness, sesh_tempo = normalized
        seq_vectors.append(torch.tensor([sesh_loudness, sesh_tempo], **kwargs))

    return dataclasses.replace(
        result,
        token_vectors=pad_sequence(token_vectors, batch_first=True),
        token_vector_idx=token_vector_idx,
        seq_vectors=torch.stack(seq_vectors, dim=0),
        max_audio_len=torch.tensor(max_audio_len, dtype=torch.long, **kwargs),
    )

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
    # torch.FloatTensor [num_tokens, *]
    token_embeddings: typing.List[torch.Tensor]

    # Slice of tokens in each sequence to be voiced
    slices: typing.List[slice]

    # The maximum audio length to generate for this text in number of frames.
    # NOTE: This must be a positive value greater or equal to one.
    max_audio_len: typing.List[int]

    # The number of annotations present
    num_anno: int

    device: torch.device = torch.device("cpu")

    # Embeddings associated with each token in each sequence
    # torch.FloatTensor [batch_size, num_tokens, *]
    token_embeddings_padded: torch.Tensor = dataclasses.field(init=False)

    # Number of tokens after `slices` is applied
    # torch.LongTensor [batch_size]
    num_tokens: torch.Tensor = dataclasses.field(init=False)

    # Tokens mask after `slices` is applied
    # torch.BoolTensor [batch_size, num_tokens]
    tokens_mask: torch.Tensor = dataclasses.field(init=False)

    # The maximum audio length for each sequence.
    # torch.LongTensor [batch_size]
    max_audio_len_tensor: torch.Tensor = dataclasses.field(init=False)

    def __post_init__(self):
        indices = [s.indices(len(t)) for s, t in zip(self.slices, self.tokens)]
        num_tokens = [b - a for a, b, _ in indices]
        num_tokens_ = torch.tensor(num_tokens, dtype=torch.long, device=self.device)
        object.__setattr__(self, "num_tokens", num_tokens_)
        object.__setattr__(self, "tokens_mask", lengths_to_mask(num_tokens, device=self.device))

        embeds = self.token_embeddings
        empty = torch.empty(0, 0, 0, device=self.device)
        stacked = empty if len(embeds) == 0 else pad_sequence(embeds, batch_first=True)
        object.__setattr__(self, "token_embeddings_padded", stacked)

        object.__setattr__(self, "max_audio_len", [max(n, 1) for n in self.max_audio_len])
        max_audio_len = torch.tensor(self.max_audio_len, device=self.device, dtype=torch.long)
        object.__setattr__(self, "max_audio_len_tensor", max_audio_len)

        self.check_invariants()

    def check_invariants(self):
        # NOTE: Double-check sizing.
        batch_size = len(self.tokens)
        max_num_tokens = max(len(t) for t in self.tokens) if len(self.tokens) > 0 else 0
        max_num_voiced_tokens = int(self.num_tokens.max()) if len(self.tokens) > 0 else 0
        assert all(len(metadata) == batch_size for metadata in self.seq_metadata)
        assert all(len(metadata) == batch_size for metadata in self.token_metadata)
        assert len(self.token_embeddings) == batch_size
        assert len(self.slices) == batch_size
        assert len(self.max_audio_len) == batch_size
        assert self.max_audio_len_tensor.shape == (batch_size,)
        assert self.token_embeddings_padded.shape[:2] == (batch_size, max_num_tokens)
        assert self.num_tokens.shape == (batch_size,)
        assert self.tokens_mask.shape == (batch_size, max_num_voiced_tokens)
        for metadata in self.token_metadata:
            assert all(len(t) == len(m) or len(m) == 0 for t, m in zip(self.tokens, metadata))
        if isinstance(self.token_embeddings, list):
            assert all(len(e) == len(t) for e, t in zip(self.token_embeddings, self.tokens))
        else:
            assert self.token_embeddings.shape[1] == max(len(seq) for seq in self.tokens)

    @property
    def anno_embeddings(self):
        # NOTE: This is determined by `preprocess` which instantiates this object.
        # TODO: Adjust the datastructure so it's easy to verify that the `token_embeddings` were
        # instantiated correctly.
        return self.token_embeddings_padded[: self.num_anno]


SpanDoc = typing.Union[spacy.tokens.span.Span, spacy.tokens.doc.Doc]


InputsWrapperTypeVar = typing.TypeVar("InputsWrapperTypeVar")
SliceAnno = typing.Tuple[slice, float]
SliceAnnos = typing.List[SliceAnno]
TokenAnnos = typing.Dict[spacy.tokens.token.Token, str]


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

    respellings: typing.List[TokenAnnos]

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
                        raise PublicValueError("The annotations must wrap words fully")
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

        # NOTE: Check that respellings are correctly formatted and wrap words entirely.
        for span_, token_annotations in zip(self.span, self.respellings):
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
        respellings = self.respellings[i].items()
        annotations = [(open_(_Schema.RESPELL, a), _idx(span, t)) for t, a in respellings]
        annotations += [(close(_Schema.RESPELL), _idx(span, t) + len(t)) for t, _ in respellings]
        annotations += [(open_(_Schema.LOUDNESS, a), s.start) for s, a in self.loudness[i]]
        annotations += [(close(_Schema.LOUDNESS), s.stop) for s, _ in self.loudness[i]]
        annotations += [(open_(_Schema.TEMPO, a), s.start) for s, a in self.tempo[i]]
        annotations += [(close(_Schema.TEMPO), s.stop) for s, _ in self.tempo[i]]
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

    def get(self, i: int):
        """Get the ith item in `self`."""
        fields = dataclasses.fields(self)
        return self.__class__(**{f.name: [getattr(self, f.name)[i]] for f in fields})

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

        respellings = {}
        for slice_, value in annotations[_Schema.RESPELL]:
            token = span.char_span(*tuple(slice_))
            if token is None or len(token) != 1:
                raise PublicValueError("Respelling must wrap a single word.")
            respellings[token[0]] = value

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
            respellings=[respellings],
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


def _embed_anno(
    length: int,
    anno: typing.List[typing.Tuple[slice, typing.Union[int, float]]],
    device: torch.device,
    idx_offset: int = 0,
    val_average: float = 0,
    val_compression: float = 1,
    avg_anno_length: int = 1,
) -> torch.Tensor:
    """Given annotations for a sequence of `length`, this returns an embedding.

    NOTE: Usually, for training, it's helpful if the data is within a range of -1 to 1. This
          function provides a `val_offset` and `val_compression` parameter to adjust the annotation
          range as needed.
    NOTE: We set the average to zero for consistency, so, if there is no annotation, it's as if
          it was annotated with the average.
    NOTE: A mask is required until enough a consistent enough interface is created. This could
          be an interface where everything is annotated, just some things are more annotated. So,
          there is never a unmasked portion.

    Args:
        length: The length of the annotated sequence.
        anno: A list of annotations.
        avg_anno_length
        idx_offset: Offset the annotation indicies.
        val_average: Offset the annotation values so that the average falls on zero.
        val_compression: Compress the annotation values so that they are easier to model.

    Returns:
        torch.FloatTensor [length, 3]
            vals: The annotated values (with offset and compression applied). A zero represents
                  no annotation.
            len_: This is the inverse of the length of the annotation. This helps the model
                  understand how "strict" the annotation is. As this goes to infinity, this goes
                  to zero, which aligns nicely with `val_average`.
            mask: This is 1 when there is an annotation and 0 when there is not.
    """
    vals = torch.zeros(length, device=device)
    mask = torch.zeros(length, device=device)
    len_ = torch.zeros(length, device=device)
    for slice_, val in anno:
        slice_ = slice(slice_.start + idx_offset, slice_.stop + idx_offset, slice_.step)
        vals[slice_] = val
        mask[slice_] = 1
        len_[slice_] = 0  # TODO: Figure out how we might handle annos of different lengths.
    vals = ((vals - val_average) / val_compression) * mask
    ret_ = torch.stack((vals, len_, mask), dim=1)
    return ret_


def _offset(annos: SliceAnnos, updates: typing.List[typing.Tuple[slice, int]]) -> SliceAnnos:
    """Adjust `annos` based on `updates` such as respellings additions."""
    slices = [s for s, _ in annos]
    return [(o, v) for o, (_, v) in zip(offset_slices(slices, updates), annos)]


def _tok_to_char_slice(span: SpanDoc, token: spacy.tokens.token.Token) -> slice:
    return slice(_idx(span, token), _idx(span, token) + len(token.text))


def preprocess(
    wrap: InputsWrapper,
    loudness_kwargs: typing.Dict,
    tempo_kwargs: typing.Dict,
    get_max_audio_length: typing.Callable[[str], int],
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
        wrap: A raw batch of data that needs to be preprocessed.
        loudness_kwargs: Key-word arguments for preprocessing loudness annotations.
        tempo_kwargs: Key-word arguments for preprocessing tempo annotations.
        get_max_audio_length: A callable for determining the maximum audio length in frames for
            a given piece of text.
        ...
    """
    num_anno = None
    inputs = Inputs([], [], [[], []], [], [], [], 0, device)
    iter_ = zip(wrap.session, wrap.span, wrap.context, wrap.loudness, wrap.tempo, wrap.respellings)
    Item = typing.Tuple[struc.Session, SpanDoc, SpanDoc, SliceAnnos, SliceAnnos, TokenAnnos]
    iter_ = typing.cast(typing.Iterator[Item], iter_)
    for sesh, span, context, loudness, tempo, respells in iter_:
        seq_metadata = [sesh[0].label, sesh, sesh[0].dialect, sesh[0].style, sesh[0].language]
        if len(inputs.seq_metadata) == 0:
            inputs.seq_metadata.extend([[] for _ in seq_metadata])
        [inputs.seq_metadata[i].append(data) for i, data in enumerate(seq_metadata)]

        tokens: typing.List[str] = []
        pronun: typing.List[Pronun] = []
        cntxt: typing.List[Context] = []
        org_tokens: typing.List[str] = []
        for tk in context:
            tokens.extend((respells[tk] if tk in respells else tk.text, tk.whitespace_))
            org_tokens.extend((tk.text, tk.whitespace_))
            pronun.extend((Pronun.RESPELLING if tk in respells else Pronun.NORMAL, Pronun.NORMAL))
            cntxt.append(Context.SCRIPT if tk in span else Context.CONTEXT)
            cntxt.append(Context.SCRIPT if tk in span and tk != span[-1] else Context.CONTEXT)

        # NOTE: Discard the trailing whitespace
        pronun, tokens, org_tokens, cntxt = pronun[:-1], tokens[:-1], org_tokens[:-1], cntxt[:-1]

        assert "".join(org_tokens) == context.text
        assert "".join(t for c, t in zip(cntxt, org_tokens) if c is Context.SCRIPT) == span.text

        chars = [c for t in tokens for c in t]
        casing = [_get_case(c) for c in chars]
        pronun = [p for t, p in zip(tokens, pronun) for _ in range(len(t))]
        cntxt: typing.List[Context] = [c for t, c in zip(tokens, cntxt) for _ in range(len(t))]
        start_char = next(i for i, c in enumerate(cntxt) if c is Context.SCRIPT)
        end_char = start_char + cntxt.count(Context.SCRIPT)

        inputs.slices.append(slice(start_char, end_char))
        # TODO: This is able to reliably determine the max audio length based on the dataset;
        # however, during inference, the user may try to push the model to slow down even further
        # using annotations. Should those be considered?
        inputs.max_audio_len.append(get_max_audio_length(span.text))
        inputs.tokens.append([c.lower() for c in chars])
        # NOTE: We merge `pronun` and `casing` into one category for performance reasons. It's
        # faster to have less unique categories. Furthermore, since casing and pronunication are
        # so prevelant in the dataset, it shoudn't have a meaningful impact on the model to have
        # these joined together.
        # TODO: Consider merging `pronun`, `casing`, `cntxt`.
        inputs.token_metadata[0].append(list(zip(pronun, casing)))
        inputs.token_metadata[1].append(cntxt)  # type: ignore

        embed = []
        for token in context:
            assert token.tensor is not None
            embed.append(np.concatenate((token.tensor, token.vector)))  # type: ignore
            embed.append(np.zeros(token.vector.shape[0] + token.tensor.shape[0]))
        embed = embed[:-1]  # NOTE: Discard the trailing whitespace
        embed = [torch.tensor(t, device=device, dtype=torch.float32) for t in embed]
        embed = torch.cat([e.unsqueeze(0).repeat(len(t), 1) for e, t in zip(embed, tokens)])

        # NOTE: Offset the loudness and tempo slices based on respellings added to text.
        respell_updates = [(_tok_to_char_slice(span, t), len(v)) for t, v in respells.items()]
        loudness = _offset(loudness, respell_updates)
        tempo = _offset(tempo, respell_updates)

        loudness_embed = _embed_anno(len(chars), loudness, device, start_char, **loudness_kwargs)
        tempo_embed = _embed_anno(len(chars), tempo, device, start_char, **tempo_kwargs)

        # TODO: Use the average loudness and tempo annotations. We should consider having
        # them as a seperate annotation, so that, the user can't accidently trick the model
        # into changing sessions.

        # loudness_embed    (torch.FloatTensor [num_tokens, 3]) (cat)
        # tempo_embed       (torch.FloatTensor [num_tokens, 3]) →
        # [num_tokens, num_anno]
        anno_embed = torch.cat((loudness_embed, tempo_embed), dim=1)
        assert num_anno is None or anno_embed.shape[1] == num_anno
        num_anno = anno_embed.shape[1]

        # anno_embed    (torch.FloatTensor [num_tokens, num_anno]) (cat)
        # embed         (torch.FloatTensor [num_tokens, embedding_size]) →
        # [num_tokens, embedding_size + num_anno]
        typing.cast(list, inputs.token_embeddings).append(torch.cat((anno_embed, embed), dim=1))

    assert num_anno is not None
    return dataclasses.replace(inputs, num_anno=num_anno)

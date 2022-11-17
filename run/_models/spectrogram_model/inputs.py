import dataclasses
import enum
import functools
import pathlib
import typing

import config as cf
import numpy as np
import spacy
import spacy.tokens
import torch
from lxml import etree
from torch.nn.utils.rnn import pad_sequence

from lib.text import XMLType
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
    # torch.FloatTensor [num_tokens, *]
    token_embeddings: typing.List[torch.Tensor]

    # Slice of tokens in each sequence to be voiced
    slices: typing.List[slice]

    # The size of the annotations added to `token_embeddings`.
    anno_mask_indices: typing.Tuple[int, ...]

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

    # A mask for each annotation
    # torch.BoolTensor [batch_size, num_tokens, len(self.anno_mask_indices)]
    anno_mask: torch.Tensor = dataclasses.field(init=False)

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

        mask = empty
        if len(self.anno_mask_indices) != 0:
            assert self.token_embeddings_padded.shape[2] > max(self.anno_mask_indices)
            indices = torch.tensor(self.anno_mask_indices, device=self.device)
            mask = torch.index_select(stacked, 2, indices)
        object.__setattr__(self, "anno_mask", mask)

        self.check_invariants()

    def check_invariants(self):
        # NOTE: Double-check sizing.
        batch_size = len(self.tokens)
        assert all(len(metadata) == batch_size for metadata in self.seq_metadata)
        assert all(len(metadata) == batch_size for metadata in self.token_metadata)
        assert len(self.token_embeddings) == batch_size
        assert len(self.slices) == batch_size
        for metadata in self.token_metadata:
            assert all(len(t) == len(m) or len(m) == 0 for t, m in zip(self.tokens, metadata))
        if isinstance(self.token_embeddings, list):
            assert all(len(e) == len(t) for e, t in zip(self.token_embeddings, self.tokens))
        else:
            assert self.token_embeddings.shape[1] == max(len(seq) for seq in self.tokens)

        # NOTE: Double-check that `anno_mask` is actually a mask.
        unique = torch.unique(self.anno_mask, sorted=True).tolist()
        assert unique == [0, 1] or unique == [0] or unique == [1] or unique == []


SpanDoc = typing.Union[spacy.tokens.span.Span, spacy.tokens.doc.Doc]


InputsWrapperTypeVar = typing.TypeVar("InputsWrapperTypeVar")
SpanAnnotation = typing.Tuple[slice, float]
SpanAnnotations = typing.List[SpanAnnotation]
TokenAnnotations = typing.Dict[spacy.tokens.token.Token, str]


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
        for name, unit, annotation_batch, min_, max_ in (
            ("Loudness", "db", self.loudness, min_loudness, max_loudness),
            ("Tempo", "seconds per character", self.tempo, min_tempo, max_tempo),
        ):
            for annotations in annotation_batch:
                if len(annotations) > 0:
                    min_seen = min(a[1] for a in annotations)
                    max_seen = max(a[1] for a in annotations)
                    message = f"{name} must be between {min_} and {max_} {unit}, got: "
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

    def _idx(self, span: SpanDoc, token: spacy.tokens.token.Token) -> int:
        """Get the character offset for `token` relative to `span`."""
        if isinstance(span, spacy.tokens.span.Span):
            return token.idx - span.start_char
        return token.idx

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
        annotations = [(open_(_Schema.LOUDNESS, a), s.start) for s, a in self.loudness[i]]
        annotations += [(close(_Schema.LOUDNESS), s.stop) for s, _ in self.loudness[i]]
        annotations += [(open_(_Schema.TEMPO, a), s.start) for s, a in self.tempo[i]]
        annotations += [(close(_Schema.TEMPO), s.stop) for s, _ in self.tempo[i]]
        respellings = self.respellings[i].items()
        idx_ = self._idx
        annotations += [(open_(_Schema.RESPELL, a), idx_(span, t)) for t, a in respellings]
        annotations += [(close(_Schema.RESPELL), idx_(span, t) + len(t)) for t, _ in respellings]
        # TODO: Create a workbook that allows us to dig into `Batch` data specifically and
        # see what is in the preprocessed and XML data, and make sure it's correct.
        annotations = sorted(annotations, key=lambda k: k[1], reverse=True)
        text = span.text
        for annotation, idx in annotations:
            text = text[:idx] + annotation + text[idx:]
        root = open_(_Schema.SPEAK, session_vocab[self.session[i]] if session_vocab else -1)
        text = f"{root}{text}{close(_Schema.SPEAK)}"
        if include_context and isinstance(span, spacy.tokens.span.Span):
            start_char = next((idx_(context, t) for t in context if t in span), 0)
            text = f"{context.text[:start_char]}{text}{context.text[start_char + len(span.text):]}"
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
                    text += typing.cast(str, elem.text).lstrip()
                elif elem.text:
                    text += elem.text
            elif event == "end":
                if elem.tag is not None and elem.tag != str(_Schema.SPEAK):
                    annotations[_Schema[elem.tag.upper()]][-1][0].append(len(text))
                if elem.tail:
                    text += elem.tail
        text = text.strip()
        assert text == span.text, "The `Span` must have the same text as the XML."
        assert session is not None

        respellings = {}
        for slice_, value in annotations[_Schema.RESPELL]:
            token = span.char_span(*tuple(slice_))
            if token is None or len(token) != 1:
                raise PublicValueError("Respelling must wrap a single word.")
            respellings[token[0]] = value

        try:
            loudness = [(s, float(v)) for s, v in annotations[_Schema.LOUDNESS]]
            tempo = [(s, float(v)) for s, v in annotations[_Schema.TEMPO]]
        except ValueError:
            raise PublicValueError("The loudness and tempo annotations must be numerical.")

        return cls(
            session=[session],
            span=[span],
            context=[span if context is None else context],
            loudness=[[(slice(*tuple(s)), v) for s, v in loudness]],
            tempo=[[(slice(*tuple(s)), v) for s, v in tempo]],
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


ANNO_MASK_INDICES = (2, 5, 6)


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
        len_[slice_] = avg_anno_length / (slice_.stop - slice_.start)
    vals = ((vals - val_average) / val_compression) * mask
    ret_ = torch.stack((vals, len_, mask), dim=1)
    return ret_


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
    inputs = Inputs([], [], [[], []], [], [], (), device)
    iter_ = zip(wrap.session, wrap.span, wrap.context, wrap.loudness, wrap.tempo, wrap.respellings)
    Item = typing.Tuple[
        struc.Session, SpanDoc, SpanDoc, SpanAnnotations, SpanAnnotations, TokenAnnotations
    ]
    iter_ = typing.cast(typing.Iterator[Item], iter_)
    for sesh, span, context, loudness, tempo, respell_map in iter_:
        seq_metadata = [sesh[0].label, sesh, sesh[0].dialect, sesh[0].style, sesh[0].language]
        if len(inputs.seq_metadata) == 0:
            inputs.seq_metadata.extend([[] for _ in seq_metadata])
        [inputs.seq_metadata[i].append(data) for i, data in enumerate(seq_metadata)]

        tokens: typing.List[str] = []
        pronun: typing.List[Pronun] = []
        cntxt: typing.List[Context] = []
        for tk in context:
            pronun.append(Pronun.RESPELLING if tk in respell_map else Pronun.NORMAL)
            tokens.append(respell_map[tk] if tk in respell_map else tk.text)
            cntxt.append(Context.SCRIPT if tk in span else Context.CONTEXT)
            pronun.append(Pronun.NORMAL)
            tokens.append(tk.whitespace_)
            cntxt.append(Context.SCRIPT if tk in span and tk != span[-1] else Context.CONTEXT)

        chars = [c for t in tokens for c in t]
        casing = [_get_case(c) for c in chars]
        pronun = [p for t, p in zip(tokens, pronun) for _ in range(len(t))]
        cntxt: typing.List[Context] = [c for t, c in zip(tokens, cntxt) for _ in range(len(t))]
        start_char = next(i for i, c in enumerate(cntxt) if c is Context.SCRIPT)
        end_char = start_char + cntxt.count(Context.SCRIPT)

        inputs.slices.append(slice(start_char, end_char))
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
        embed = [torch.tensor(t, device=device, dtype=torch.float32) for t in embed]
        embed = torch.cat([e.unsqueeze(0).repeat(len(t), 1) for e, t in zip(embed, tokens)])
        vector_embed_mask = torch.ones(embed.shape[0], 1, device=device)

        loudness_embed = _embed_anno(len(chars), loudness, device, start_char, **loudness_kwargs)
        tempo_embed = _embed_anno(len(chars), tempo, device, start_char, **tempo_kwargs)

        # loudness_embed    (torch.FloatTensor [num_tokens, 3]) (cat)
        # tempo_embed       (torch.FloatTensor [num_tokens, 3]) (cat)
        # vector_embed_mask (torch.FloatTensor [num_tokens, 1]) (cat)
        # embed             (torch.FloatTensor [num_tokens, embedding_size]) →
        # [num_tokens, embedding_size + 7]
        embed = torch.cat((loudness_embed, tempo_embed, vector_embed_mask, embed), dim=1)
        typing.cast(list, inputs.token_embeddings).append(embed)

    return dataclasses.replace(inputs, anno_mask_indices=ANNO_MASK_INDICES)

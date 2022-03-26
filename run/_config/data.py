import copy
import logging
import pathlib
import typing

import config as cf

import lib
import run
from run.data import _loader
from run.data._loader import Language, Passage, Span

logger = logging.getLogger(__name__)

# NOTE: This is useful for one-off evaluation.
DEFAULT_SCRIPT = (
    "Your creative life will evolve in ways that you can’t possibly imagine. Trust"
    " your gut. Don’t overthink it. And allow yourself a little room to play."
)

DATASETS = copy.copy(_loader.DATASETS)
# NOTE: Elliot and Elizabeth has unannotated character portrayals.
del DATASETS[_loader.english.ELLIOT_MILLER]
del DATASETS[_loader.english.ELIZABETH_KLETT]

DEV_SPEAKERS = _loader.WSL_DATASETS.copy()
# NOTE: The `MARI_MONGE__PROMO` dataset is too short for evaluation, at 15 minutes long.
del DEV_SPEAKERS[_loader.english.MARI_MONGE__PROMO]
# NOTE: The `ALICIA_HARRIS`, `JACK_RUTKOWSKI`, and `SAM_SCHOLL` datasets are duplicate datasets.
# There is an improved version of their datasets already in `dev_speakers`.
del DEV_SPEAKERS[_loader.english.ALICIA_HARRIS]
del DEV_SPEAKERS[_loader.english.JACK_RUTKOWSKI]
del DEV_SPEAKERS[_loader.english.SAM_SCHOLL]
# NOTE: The `BETH_CAMERON__CUSTOM` dataset isn't included in the studio.
del DEV_SPEAKERS[_loader.english.BETH_CAMERON__CUSTOM]

for dataset in [DEV_SPEAKERS, DATASETS]:
    # NOTE: The following custom datasets are poor quality and should be excluded.
    del dataset[_loader.english.HOUR_ONE_NBC__BB_CUSTOM_VOICE]
    del dataset[_loader.english.VIACOM__CUSTOM_VOICE]
    del dataset[_loader.english.UNEEQ__ASB_CUSTOM_VOICE]
    # NOTE: The alignments don't match up with the scripts.
    del dataset[_loader.english.UNEEQ__ASB_CUSTOM_VOICE_COMBINED]

DEV_SPEAKERS = set(DEV_SPEAKERS.keys())


def _include_passage(
    passage: Passage, root: pathlib.Path, language: typing.Optional[Language] = None
) -> bool:
    """Return `True` iff `passage` should be included in the dataset."""
    repr_ = f"{passage.__class__.__name__}("
    repr_ += f"{passage.audio_file.path.relative_to(root)},"
    repr_ += f" {(passage.script[:50] + '...') if len(passage.script) > 50 else passage.script})"

    if language is not None and passage.speaker.language != language:
        return False

    if len(passage.alignments) == 0:
        logger.warning("%s has zero alignments.", repr_)
        return False

    if len(passage.speech_segments) == 0:
        logger.warning("%s has zero speech segments.", repr_)
        return False

    span = passage[:]
    if span.audio_length == 0.0:
        logger.warning("%s has no aligned audio.", repr_)
        return False

    if len(span.script) == 0:
        logger.warning("%s has no aligned text.", repr_)
        return False

    # NOTE: Filter out passages(s) that don't have a lower case character because it'll make
    # it difficult to classify initialisms.
    if not any(c.islower() for c in passage.script):
        return False

    # TODO: Filter out Mary Ann from the dataset instead of filtering the related books.
    # NOTE: Filter out Midnight Passenger because it has an inconsistent acoustic setup compared to
    # other samples from the same speaker.
    # NOTE: Filter out the North & South book because it uses English in a way that's not consistent
    # with editor usage, for example: "To-morrow, you will-- Come back to-night, John!"
    books = (
        _loader.english.m_ailabs.MIDNIGHT_PASSENGER,
        _loader.english.m_ailabs.NORTH_AND_SOUTH,
    )
    metadata = passage.other_metadata
    if metadata is not None and "books" in metadata and (metadata["books"] in books):
        return False

    return True


def _include_span(span: Span):
    """Return `True` iff `span` should be included in the dataset.

    TODO: The dataset metrics show that 2% of Heather's dataset still has pauses longer than 1s.
    Can we filter them out in accordance to `too_long_pause_length`?
    TODO: How can we filter out all non-standard words that haven't been normalized, yet? We could
    normalize the script before hand, removing all non-standard words. Afterwards, we can verify
    with Google STT that it matches the voice over.
    TODO: The character "." is ambigious. It is sometimes prounced "dot" and sometimes it's silent.
    There may be some inconsistency between eSpeak and the voice over with regards to ".".
    """
    script = str(span.spacy_with_context(**cf.get()))

    if "<" in script or ">" in script:
        return False

    # NOTE: Filter out any passage(s) with a slash because it's ambigious. It's not obvious if
    # it should be silent or verbalized.
    if "/" in script or "\\" in script:
        return False

    # NOTE: Filter out any passage(s) with digits because the pronunciation is fundamentally
    # ambigious, and it's much easier to handle this case with text normalization.
    # NOTE: See performance statistics here: https://stackoverflow.com/a/31861306/4804936
    if lib.text.has_digit(script):
        return False

    # NOTE: `Span`s which end with a short, or fast `Span`, tend to be error prone.
    is_not_aligned = lambda s: s.audio_length < 0.2 or (s.audio_length / len(s.script)) < 0.04
    if is_not_aligned(span[0]) or is_not_aligned(span[-1]):
        return False

    if _loader.has_a_mistranscription(span):
        return False

    return True


def configure():
    """Configure modules that process data, other than audio."""
    # TODO: Remove `BETH_CAMERON__CUSTOM` from the `WSL_DATASETS` groups because it has it's own
    # custom script.
    groups = [set(_loader.WSL_DATASETS.keys())]
    # NOTE: For other datasets like M-AILABS and LJ, this assumes that there is no duplication
    # between different speakers.
    groups += [{s} for s in _loader.DATASETS.keys() if s not in _loader.WSL_DATASETS]
    config = {
        run._utils.get_dataset: cf.Args(
            datasets=DATASETS,
            include_psge=_include_passage,
            handle_psge=lib.utils.identity,
        ),
        run._utils.split_dataset: cf.Args(
            groups=groups, dev_speakers=DEV_SPEAKERS, approx_dev_len=30 * 60, min_sim=0.9
        ),
        run.data._loader.data_structures.Span.spacy_with_context: cf.Args(max_words=20),
        run._utils.SpanGenerator: cf.Args(max_seconds=15, include_span=_include_span),
    }
    cf.add(config)

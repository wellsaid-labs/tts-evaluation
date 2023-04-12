import copy
import functools
import itertools
import logging
import math
import typing

import config as cf
import numpy as np

import lib
import run
from run.data import _loader
from run.data._loader import structures as struc

logger = logging.getLogger(__name__)

# NOTE: This is useful for one-off evaluation.
DEFAULT_SCRIPT = (
    "Your creative life will evolve in ways that you can't possibly imagine. Trust"
    " your gut. Don't overthink it. And allow yourself a little room to play."
)

DATASETS = copy.copy(_loader.DATASETS)
# NOTE: Elliot and Elizabeth has unannotated character portrayals.
del DATASETS[_loader.english.m_ailabs.ELLIOT_MILLER]
del DATASETS[_loader.english.m_ailabs.ELIZABETH_KLETT]
# NOTE: Filter out Mary Ann from the dataset because of her two books, which include:
# - Midnight Passenger because it has an inconsistent acoustic setup compared to other samples from
# the same speaker.
# - North & South book because it uses English in a way that's not consistent with editor usage,
# for example: "To-morrow, you will-- Come back to-night, John!"
del DATASETS[_loader.english.m_ailabs.MARY_ANN]
# NOTE: The alignments don't sound-a-like, in these datasets.
del DATASETS[_loader.portuguese.librivox.RND__LIBRIVOX__FELIPE_PT]
del DATASETS[_loader.portuguese.librivox.RND__LIBRIVOX__LENI_PT]
del DATASETS[_loader.portuguese.librivox.RND__LIBRIVOX__MIRAMONTES_PT]
del DATASETS[_loader.portuguese.librivox.RND__LIBRIVOX__SANDRALUNA_PT]
# NOTE: The `AVA_M`, `KAI_M`, and `WADE_C` datasets are duplicate datasets.
# There is an improved version of their datasets already in `DATASETS`.
del DATASETS[_loader.english.wsl.AVA_M]
del DATASETS[_loader.english.wsl.KAI_M]
del DATASETS[_loader.english.wsl.WADE_C]

# TODO: Remove any non-production datasets from `WSL_DATASETS` so we don't evaluate on them.
DEV_SPEAKERS = _loader.WSL_DATASETS.copy()
# NOTE: The `MARI_MONGE__PROMO` and `MARCUS_G__CONVO` datasets are too short for evaluation,
# at 15 minutes and 60 minutes long, respectively.
del DEV_SPEAKERS[_loader.english.wsl.MARI_MONGE__PROMO]
del DEV_SPEAKERS[_loader.english.wsl.MARCUS_G__CONVO]
# NOTE: The `RAMONA_J__CUSTOM` dataset isn't included in the studio.
del DEV_SPEAKERS[_loader.english.wsl.RAMONA_J__CUSTOM]
# NOTE: Elizabeth's dataset is low quality & might be fixed or re-recorded. Tobin's did not differ
# from his Narration enough to be considered new styles & both datasets were again quick &
# inconsistent delivery.
del DEV_SPEAKERS[_loader.english.wsl.TOBIN_A__CONVO]
del DEV_SPEAKERS[_loader.english.wsl.TOBIN_A__PROMO]
del DEV_SPEAKERS[_loader.english.wsl.ELIZABETH_U]

for dataset in [DEV_SPEAKERS, DATASETS]:
    # NOTE: The following custom datasets are poor quality and should be excluded.
    del dataset[_loader.english.wsl.HOUR_ONE_NBC__BB_CUSTOM_VOICE]
    del dataset[_loader.english.wsl.VIACOM__CUSTOM_VOICE]
    del dataset[_loader.english.wsl.UNEEQ__ASB_CUSTOM_VOICE]
    # NOTE: The alignments don't match up with the scripts.
    del dataset[_loader.english.wsl.UNEEQ__ASB_CUSTOM_VOICE_COMBINED]
    # NOTE: The alignments don't sound-a-like, in these datasets.
    del dataset[_loader.portuguese.wsl.FIVE_NINE__CUSTOM_VOICE__PT_BR]
    del dataset[_loader.spanish.wsl.FIVE_NINE__CUSTOM_VOICE__ES_CO]

DEV_SPEAKERS = set(DEV_SPEAKERS.keys())
# NOTE: It's generally useful to evaluate the model on a dictionary dataset, that has a variety
# of words and acronyms.
DEV_SPEAKERS.add(_loader.english.dictionary.GCP_SPEAKER)


def _include_passage(passage: struc.Passage) -> bool:
    """Return `True` iff `passage` should be included in the dataset."""
    repr_ = f"{passage.__class__.__name__}({passage.speaker.label}, {passage.session.label}, "
    repr_ += f"{(passage.script[:25] + '...') if len(passage.script) > 25 else passage.script})"

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
    # NOTE: Ensure that single word initialism scripts are preserved such as those in a
    # pronunciation dictionary.
    if passage.speaker.style is not struc.Style.DICT and passage.script.isupper():
        logger.warning("%s is all uppercase.", repr_)
        return False

    return True


def _include_span(span: struc.Span):
    """Return `True` iff `span` should be included in the dataset.

    TODO: The dataset metrics show that 2% of Heather's dataset still has pauses longer than 1s.
          Can we filter them out in accordance to `too_long_pause_length`?
    TODO: How can we filter out all non-standard words that haven't been normalized, yet? We could
          normalize the script before hand, removing all non-standard words. Afterwards, we can
          verify with Google STT that it matches the voice over.
    TODO: The character "." is ambiguous. It is sometimes prounced "dot" and sometimes it's silent.
          There may be some inconsistency between eSpeak and the voice over with regards to ".".
    TODO: Add a filter using `get_max_audio_length`, it'll help filter out clips with excessive
          silence. It'll also make sure that clips actually subscribe to this limit.
    """
    script = str(span.spacy_context(**cf.get()))

    if "<" in script or ">" in script or "&" in script:
        return False

    # NOTE: Questions in `NARR` style voices tend to fall flat, largely due to the content
    # the voice actors are reading. This behavior is unexpected for customers, so we filtered
    # out these questions.
    if "?" in script and span.speaker.style is struc.Style.OG_NARR:
        return False

    # NOTE: Filter out any passage(s) with a slash because it's ambiguous. It's not obvious if
    # it should be silent or verbalized.
    if "/" in script or "\\" in script:
        return False

    # NOTE: Filter out any passage(s) with digits because the pronunciation is fundamentally
    # ambiguous, and it's much easier to handle this case with text normalization.
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


def _include_annotation(annotation: struc.Alignment):
    """Return `True` iff `annotation` should be included in the dataset."""
    # TODO: This criteria matches the the above `Span` criteria, and could be fleshed out more.
    # Generally, we've found that `Alignment`s which are too fast or too short, are error prone.
    if annotation.audio_len < 0.2:
        return False

    if annotation.audio_len / annotation.script_len < 0.04:
        return False

    return True


def _get_tempo_annotation(text: str, audio_len: float, bucket_size: float) -> float:
    """Get a tempo annotation in actual length vs expected length.

    Args:
        ...
        audio_len: The audio length in seconds.
        bucket_size: The bucket size for rounding in seconds.
    """
    if audio_len == 0:
        return math.nan

    avg = run._config.lang.get_avg_audio_length(text)
    return lib.utils.round_(avg / audio_len, bucket_size)


def _get_loudness_annotation(
    audio: typing.Union[np.ndarray, typing.List[np.ndarray]],
    sample_rate: int,
    block_size: float,
    precision: int,
    **kwargs,
) -> typing.Optional[float]:
    """Get the loudness in LUFS for `audio`.

    NOTE: `integrated_loudness` filters out quiet sections from the loudness computations.
    NOTE: The minimum audio length for calculating loudness is the `block_size` which is typically
          around 400ms.
    TODO: Let's investigate how well this matches with folks expectations of loudness, the LUFS
          algorithm isn't built to match perception accross the entire range. LUFS uses
          K-weighting which was initially built for music/radio.

    Args:
        ...
        precision: The number of decimal places to round LUFS.

    Returns: The loudness in LUFS with a range of 0 to -70 LUFS in alignment with ITU-R BS.1770-4.
        This returns `None` if the loudness cannot be computed.
    """
    sec_to_sample_ = functools.partial(lib.audio.sec_to_sample, sample_rate=sample_rate)

    if isinstance(audio, list):
        # TODO: Test if zero padding affects the loudness computation; whilst this would be
        # disheartening, it shouldn't have a big impact, either way. Hopefully, the algorithm
        # is invariant to padding.
        padding = np.zeros((sec_to_sample_(block_size),))
        audio = np.concatenate([n for a in audio for n in (a, padding)][:-1])

    meter = lib.audio.get_pyloudnorm_meter(sample_rate, block_size=block_size, **kwargs)
    if audio.shape[0] >= sec_to_sample_(block_size):
        loudness = round(float(meter.integrated_loudness(audio)), precision)
        # NOTE: This algorithm returns negative infinity if the loudness is less than -70 LUFS. We
        # return -70 LUFS instead to keep the output finite.
        # NOTE: This number is not parameterized because this specific number is specified in
        # the LUFS algorithm specification, ITU-R BS.1770-4.
        # NOTE: The loudness algorithm can sometimes overflow and return strange values that are
        # significantly outside of the range like in:
        # https://github.com/csteinmetz1/pyloudnorm/issues/42
        loudness = -70.0 if math.isinf(loudness) and loudness < 0 else loudness
        assert loudness >= -70 and loudness <= 0
        assert isinstance(loudness, float)
        return loudness
    return None


def configure(overwrite: bool = False):
    """Configure modules that process data, other than audio."""
    cf.add({_get_tempo_annotation: cf.Args(bucket_size=0.05)}, overwrite)

    # TODO: Remove `BETH_CAMERON__CUSTOM` from the `WSL_DATASETS` groups because it has it's own
    # custom script.
    groups = [set(itertools.chain(_loader.WSL_DATASETS.keys(), _loader.RND_DATASETS.keys()))]
    # NOTE: For other datasets like M-AILABS and LJ, this assumes that there is no duplication
    # between different speakers.
    groups += [{s} for s in _loader.DATASETS.keys() if s not in groups[0]]
    config = {
        run._utils.get_unprocessed_dataset: cf.Args(datasets=DATASETS),
        run._utils.get_dataset: cf.Args(datasets=DATASETS, include_passage=_include_passage),
        # NOTE: Set `approx_dev_len` to 30 minutes for a consistent amount of dev data per speaker,
        # guesstimated to be a sufficient quantity to capture enough variety in each voice.
        # NOTE: Set `min_split_passages` to 3 passages, guesstimated to provide enough passage
        # variety of different content topics.
        run._utils.split_dataset: cf.Args(
            groups=groups,
            dev_speakers=DEV_SPEAKERS,
            approx_dev_len=25 * 60,
            min_sim=0.95,
            min_split_passages=3,
        ),
        run.data._loader.structures.Span.spacy_context: cf.Args(max_words=20),
        run._utils.SpanGenerator: cf.Args(max_seconds=15, include_span=_include_span),
        # NOTE: `min_no_intervals_prob` was set at 10% to ensure the model is exposed to some
        # data that has no annotations; however, our preference is for the model to train with
        # more annotations because it should "stabalize" it. As in, the model would not need to
        # guess as much which creates an easier training environment.
        # NOTE: We gueestimated that users would have around 3 annotations per clip in Studio.
        run.train.spectrogram_model._data._random_nonoverlapping_alignments: cf.Args(
            min_no_intervals_prob=0.1,
            avg_alignments=3,
            include_annotation=_include_annotation,
        ),
        run.data._loader.structures._process_sessions: cf.Args(
            get_loudness=cf.partial(_get_loudness_annotation),
            get_tempo=cf.partial(_get_tempo_annotation),
        ),
        run.train.spectrogram_model._data._get_tempo_annotation: cf.Args(
            get_anno=cf.partial(_get_tempo_annotation)
        ),
        run.train.spectrogram_model._data._get_loudness_annotation: cf.Args(
            get_anno=cf.partial(_get_loudness_annotation)
        ),
        # NOTE: We expect users to respell approx 5 - 10% of words.
        run.train.spectrogram_model._data._random_respelling_annotations: cf.Args(prob=0.1),
    }
    cf.add(config, overwrite)

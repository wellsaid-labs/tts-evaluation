import copy
import itertools
import logging

import config as cf

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
# NOTE: The `MARI_MONGE__PROMO` dataset is too short for evaluation, at 15 minutes long.
del DEV_SPEAKERS[_loader.english.wsl.MARI_MONGE__PROMO]
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
    repr_ = f"{passage.__class__.__name__}({passage.speaker.label}, {passage.session[1]}, "
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
    normalize the script before hand, removing all non-standard words. Afterwards, we can verify
    with Google STT that it matches the voice over.
    TODO: The character "." is ambigious. It is sometimes prounced "dot" and sometimes it's silent.
    There may be some inconsistency between eSpeak and the voice over with regards to ".".
    """
    script = str(span.spacy_context(**cf.get()))

    if "<" in script or ">" in script or "&" in script:
        return False

    # NOTE: Questions in `NARR` style voices tend to fall flat, largely due to the content
    # the voice actors are reading. This behavior is unexpected for customers, so we filtered
    # out these questions.
    if "?" in script and span.speaker.style is struc.Style.OG_NARR:
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


def _include_annotation(annotation: struc.Alignment):
    """Return `True` iff `annotation` should be included in the dataset."""
    audio_len = annotation.audio[1] - annotation.audio[0]
    if audio_len < 0.2:
        return False

    script_len = annotation.script[1] - annotation.script[0]
    if audio_len / script_len < 0.04:
        return False

    return True


ENGLISH_TEST_CASES = [
    # NOTE: These statements have a mix of heteronyms, initialisms, hard words (locations,
    # medical terms, technical terms), etc for testing pronunciation.
    "For more updates on covid nineteen, please contact us via the URL at the bottom of the "
    "screen, or visit our office in Seattle at the address shown here.",
    "I've listed INTJ on my resume because it's important for me that you understand how I "
    "conduct myself in stressful situations.",
    "The website is live and you can access your records via the various APIs slash URLs or use "
    "the Studio as an alternate avenue.",
    "The nurses will resume the triage conduct around the oropharyngeal and test for "
    "tachydysrhythmia to ensure the patient lives another day.",
    "Access your clusters using the Kubernetes API. You can alternate between the CLI and the "
    "web interface.",
    "Live from Seattle, it's AIQTV, with the governor's special address on the coronavirus. Don't "
    "forget to record this broadcast for viewing later.",
    "Let's add a row on our assay tracking sheet so we can build out the proper egress "
    "measurements.",
    "Hello! Can you put this contractor into a supervisory role?",
    # NOTE: These test various initialisms
    "Each line will have GA Type as Payment, Paid Amount along with PAC, and GA Code.",
    "Properly use and maintain air-line breathing systems and establish a uniform procedure "
    "for all employees, for both LACC and LCLA contractors, to follow when working jobs that "
    "require the use of fresh air.",
    "QCBS is a method of selecting transaction advisors based on both the quality of their "
    "technical proposals and the costs shown in their financial proposals.",
    "HSPs account for fifteen to twenty percent of the population.",
    "We used to have difficulty with AOA and AMA, but now we are A-okay.",
    "As far as AIs go, ours is pretty great!",
    # NOTE: These questions each have a different expected inflection.
    "If you can instantly become an expert in something, what would it be?",
    "What led to the two of you having a disagreement?",
    "Why do some words sound funny to us?",
    "What are your plans for dealing with it?",
    "There may be times where you have to RDP to a node and manually collect logs for some "
    "reason. So, another question you may have is, exactly where on disk are all these logs?",
    "How common are HSPs?",
    "If you could rid the world of one thing, what would it be?",
    "What childish things do you still do as an adult?",
    "If you were to perform in the circus, what would you do?",
    # NOTE: All these questions should have an upward inflection at the end.
    "Have you ever hidden a snack so that nobody else would find it and eat it first?",
    "Can fish see air like we see water?",
    "Are you a messy person?",
    "Did you have cats growing up?",
    "Do you consider yourself an adventurous person?",
    "Do you have any weird food combos?",
    "Do you respond to texts fast?",
    "Have you ever been stalked by an animal that later became your pet?",
    "If you have made it this far, do you relate to any of these signs? Are you a highly "
    "sensitive person?",
    "Have you started, but not found success, with a platform requiring monthly payments?",
    "When deciding between organic and non-organic coffees, is the price premium worth it?",
    "Can you make yourself disappear?",
    "Do mice really eat cheese?",
    "Do you believe in any conspiracy theories?",
    "Have elves always lived at the North Pole?",
    "Have you ever been on the radio?",
    "Have you ever done something embarrassing in front of the office CCTV cameras?",
    "In your opinion, are giant spiders worse than giant chickens?",
    "What is the process for making your favorite dish?",
    "Would you like to be part of the UK Royal Family?",
    "Did you ever try DIY projects?",
    "Can people from NASA catch the flu?",
    "Do you watch ESPN at night?",
    "Will AI replace humans?",
    "Can our AI say AI?",
    # NOTE: Test cases with a variety of lengths, respellings, and punctuation marks.
    "WellSaid Labs.",
    "Livingroom",
    "Ophthalmologist",
    "ACLA",
    "ACLA.",  # NOTE: `ACLA` sometimes gets cut-off, this is a test to see how a period affects it.
    "NASA",
    "Why?",
    'Ready to find out ""more""?',
    "Thisss isrealy awhsome.",
    "Topic two:     Is an NRA right for my rate?.",
    'Internet Assigned Numbers Authority ("""I-eigh n Eigh""")',
    '"""G-E-ran""" is an abbreviation for GSM EDGE',
    "epidermolysis bullosa (ep-ih-dur-MOL-uh-sis buhl-LOE-sah) (epi-dermo-lysiss) is a group of",
    "Harry lay in his dark cupboard much later, wishing he had a watch. He didn't know what time "
    "it was and he couldn't be sure the Dursleys were asleep yet. Until they were, he couldn't "
    "risk sneaking to the kitchen for some food. He'd lived with the Dursleys almost ten years, "
    "ten miserable years, as long as he could remember, ever since he'd been a baby and his "
    "parents had died in that car crash. He couldn't remember being in the car when his parents "
    "had died. Sometimes, when he strained his memory during long hours in his cupboard, he came "
    "up with a strange vision: a blinding flash of green light and a burning pain on his "
    "forehead. This, he supposed, was the crash, though he couldn't imagine where all the green "
    "light came from. He couldn't remember his parents at all. His aunt and uncle never spoke "
    "about them, and of course he was forbidden to ask questions. There were no photographs of "
    "them in the house. When he had been younger, Harry had dreamed and dreamed of some unknown "
    "relation coming to take him away, but it had never happened; the Dursleys were his only "
    "family. Yet sometimes he thought (or maybe hoped) that strangers in the street seemed to "
    "know him. Very strange strangers they were, too.",
    # NOTE: Test respellings
    # TODO: Adjust respellings based on latest conventions.
    "I see in “Happening at <respell value='se-FOHR-u'>Sephora</respell>” I have two new brands"
    "requesting store-led events for the same day.",
    "Welcome to the <respell value='su-LAHR-es'>Solares</respell> Injury and Illness Prevention "
    "Program Training.",
    "The <respell value='pur-AY-toh'>Pareto</respell> principle was named after Italian economist "
    "Vilfredo <respell value='pu-RAY-toh'>Pareto</respell>.",
    "We would like to nominate <respell value='AY-vu'>Avu</respell> for her phenomenal "
    "recordings.",
    "To use your self-help AI, please enable the Affirmations feature on the "
    "<respell value='KAHN-sohl'>console</respell> so that you can "
    "<respell value='kuhn-SOHL'>console</respell> yourself.",
    "Too much sand? Tired of cacti? <respell value='dee-ZURT'>desert</respell> the "
    "<respell value='DEZ-urt'>desert</respell> now, with caravan adventures!",
    "If you want to get the good food at the <respell value='bu-FAY'>buffet</respell>, you have "
    "to be willing to "
    "<respell value='BUF-et'>buffet</respell> and punch your way to the front of the line.",
    "Does <respell value='BEE-u-loh-ZHEEK'>biologique</respell> "
    "<respell value='ru-SHURSH'>recherche</respell> really work?",
    # NOTE: Test v10 regressions
    # NOTE: Respellings are formatted like they were inputted in v10
    # - Difficult acronyms
    "It took six Ph.Ds to design a VCR a five-year-old could use.",
    # - "Cape Cod" was repeated
    "It is ironic that today's least "
    "<respell value='PAH-pyuh-lay-tuhd'>|\\PAH\\pyuh\\lay\\tuhd\\|</respell> town on Cape Cod",
    # - Short sentences were cut off
    "Taking sides early - I feel like... I feel like that's a recipe for disaster. It is.",
    "manager. Egan",
    "then walked away without taking any questions. Wow,",
    "Thanks! For..",
    "using your ears. Why?",
    "Yes. Are you ready to play? Yeah.",
    # - This question generated a long silence, after "morning"
    "Can you tell me more about what happened that morning?",
    # - This word was pronounced incorrectly
    "anemone",
    # - This word caused the model to overflow
    "<respell value='po-lahn-co'>|\\po\\lahn\\co|</respell>",
    "<respell value='fran-SIH-skoh'>|\\fran\\SIH\\skoh|</respell>",
]
TEST_CASES = [(struc.Language.ENGLISH, t) for t in ENGLISH_TEST_CASES]


def configure(overwrite: bool = False):
    """Configure modules that process data, other than audio."""
    # TODO: Remove `BETH_CAMERON__CUSTOM` from the `WSL_DATASETS` groups because it has it's own
    # custom script.
    groups = [set(itertools.chain(_loader.WSL_DATASETS.keys(), _loader.RND_DATASETS.keys()))]
    # NOTE: For other datasets like M-AILABS and LJ, this assumes that there is no duplication
    # between different speakers.
    groups += [{s} for s in _loader.DATASETS.keys() if s not in groups[0]]
    config = {
        run._utils.get_dataset: cf.Args(
            datasets=DATASETS,
            include_psge=_include_passage,
            handle_psge=lib.utils.identity,
        ),
        run._utils.split_dataset: cf.Args(
            groups=groups, dev_speakers=DEV_SPEAKERS, approx_dev_len=30 * 60, min_sim=0.9
        ),
        run.data._loader.structures.Span.spacy_context: cf.Args(max_words=20),
        run._utils.SpanGenerator: cf.Args(max_seconds=15, include_span=_include_span),
        run.train._utils.process_select_cases: cf.Args(
            cases=TEST_CASES, speakers=DEV_SPEAKERS, num_cases=15
        ),
        # NOTE: `min_no_intervals_prob` was set at 10% to ensure the model is exposed to some
        # data that has no annotations; however, our preference is for the model to train with
        # more annotations because it should "stabalize" it. As in, the model would not need to
        # guess as much which creates an easier training environment.
        # NOTE: We gueestimated that users would have around 3 annotations per clip in Studio.
        # NOTE: We guesstimated that the average interval length is likely going to be around
        # 2 words which translates to 3 alignments. This parameter helps ensure that we don't
        # get examples annotated only with short annotations.
        run.train.spectrogram_model._data._random_nonoverlapping_alignments: cf.Args(
            min_no_intervals_prob=0.1,
            avg_alignments=3,
            min_avg_interval_length=3,
            include_annotation=_include_annotation,
        ),
        run.train.spectrogram_model._data._random_tempo_annotations: cf.Args(precision=2),
        # NOTE: We expect users to respell approx 5 - 10% of words.
        run.train.spectrogram_model._data._random_respelling_annotations: cf.Args(prob=0.1),
    }
    cf.add(config, overwrite)

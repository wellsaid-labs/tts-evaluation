""" A workbook to generate audio for evaluating various test cases.

TODO: Create a script that downloads multiple checkpoints at various points, generates scripts
      with them, and produces a zip file. We can use `disk/envs` to get the information I need
      to generating something like this.
TODO: Instead of using random speakers and sessions, let's consider using the choosen session
      and speakers in `deploy.sh`. Those will be deployed, anyways.
TODO: Implement `batch_griffin_lim_tts` to support batch generation, speeding up this script.

Usage:
    $ PYTHONPATH=. streamlit run run/review/tts/test_cases.py --runner.magicEnabled=false
"""
import random
import typing
from functools import partial

import config as cf
import numpy as np
import streamlit as st
from streamlit.delta_generator import DeltaGenerator
from torchnlp.random import fork_rng

import lib
import run
from lib.environment import PT_EXTENSION, load
from lib.text import XMLType
from run._config import DEFAULT_SCRIPT, SPECTROGRAM_MODEL_EXPERIMENTS_PATH
from run._config.data import _get_loudness_annotation, _get_tempo_annotation
from run._models.spectrogram_model import SpectrogramModel
from run._streamlit import (
    audio_to_web_path,
    st_download_files,
    st_html,
    st_select_path,
    web_path_to_url,
)
from run._tts import griffin_lim_tts
from run.data._loader import Speaker
from run.data._loader.english import wsl

V10_REGRESSIONS = [
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
    # ("Wii")
    "The Wii gaming console.",
    # Paige, Jarvis ("STOP")
    "This needs to STOP now.",
    # Ava ("queue")
    "We were forced to stand in a queue.",
    # Ava ("demo")
    "She will be showing a demo of the company's new alarm system.",
    # Donna, Wade ("Meet" dropping t sound)
    "We arranged to meet for lunch.",
    # Tilda, Lee, Vanessa, Jeremy, Nicole ("ChatGPT")
    "ChatGPT is a powerful chatbot developed by OpenAI that uses machine learning to generate \
      human-like responses to user input.",
    "There are several benefits to using text-to-speech with ChatGPT.",
    "Another benefit to using text-to-speech with ChatGPT is that it can make the chatbots' \
      responses sound more natural and human-like.",
    "One way to enhance the user experience with ChatGPT is by using a text to speech system to \
      convert the chatbot’s responses from text to speech, allowing users to hear the chatbot’s \
        responses instead of reading them.",
    # Phone numbers
    "My phone number is 723-5670",
    # Parenthesis
    "However when part of my script talked about a 401(k) plan and the voiceover pronounced it \
      four hundred and wonk",
    "I work for a financial industry and am making lots of videos talking about 401(k)s, 457(b)s, \
      and so on.",
]

V10_ACCENTURE = [
    # NOTE Slack Reference: https://wellsaidlabs.slack.com/archives/C0149LB6LKX/p1671134275497839
    # Wade ("EVV" pronounced with inconsistent speed)
    "Step 2: The EVV vendor reviews the EVV Provider Onboarding Form and confirms all required \
      fields are complete and accurate.",
    "Within one business day of receipt, the EVV vendor will send an email to the signature \
      authority and Program Provider/FMSA EVV System Administrator listed on the form to \
        acknowledge receipt of the EVV Provider Onboarding Form.",
    "The EVV vendor will advise that the submitted form is under review and the contact information\
      for the Program Provider/FMSA EVV System Administrator on the form will be used to contact \
        the program provider or FMSA to begin the EVV Provider Onboarding Process.",
]

V10_SYNTHESIA = [
    # NOTE Slack Reference: https://wellsaidlabs.slack.com/archives/C0149LB6LKX/p1673021131011249
    # "AI", "Xelia", "studio" requesting v10 downgrade to v9
    "I’m an AI avatar",
    "Mindblowing AI tools you’ve never heard of",
    "We are happy to support tools supporting Xelia grows",
    "Here’s a quick overview of our studio platform",
]

V10_EMAIL = [
    "hello@wellsaidlabs.com.",
    "hello123@wellsaidlabs.com.",
    "hello-123@wellsaidlabs.com.",
    "hello_123@wellsaidlabs.com.",
]

VARIOUS_INITIALISMS = [
    "Each line will have GA Type as Payment, Paid Amount along with PAC, and GA Code.",
    "Properly use and maintain air-line breathing systems and establish a uniform procedure "
    "for all employees, for both LACC and LCLA contractors, to follow when working jobs that "
    "require the use of fresh air.",
    "QCBS is a method of selecting transaction advisors based on both the quality of their "
    "technical proposals and the costs shown in their financial proposals.",
    "HSPs account for fifteen to twenty percent of the population.",
    "We used to have difficulty with AOA and AMA, but now we are A-okay.",
    "As far as AIs go, ours is pretty great!",
]

QUESTIONS_WITH_UPWARD_INFLECTION = [
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
]

QUESTIONS_WITH_VARIED_INFLECTION = [
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
    "Who is eligible for reimbursement?",
]

RESPELLINGS = [
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
]

HARD_SCRIPTS = [
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
]

HARD_SCRIPTS_2 = [
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
]

V10_AUDIO_QUALITY_DISTORTION = [
    # 05/2021: Tobin (Alison: Noticeable in datasets, but not in model output)
    "Tobin’s “breathing” events will become highly distorted approximately 80 percent of the time,\
       especially when using samples of 300-400 words.",
    # 11/2022: Ramona (Alison: Noticeable in datasets, but not in model output)
    "We hear distortion in the audio file, a hissing sound, particularly in Romana J Avatar.",
]

V10_AUDIO_QUALITY_BREATHING = [
    # 08/2020: Isabel
    "Isabel seems to struggle with more than one sentence. There is a bug with her breathing and \
      the end of sentences",
    # 2020: Kai (Alison: Not noticeable in datasets, or model output)
    # "I use Kai the most, however, I don’t always enjoy editing out all his breaths before each \
    #   sentence, but still gets used most frequently.",
    # 2020-2022: Jeremy
    # "Jeremy's voice is great- perfect speed and inflection in most cases.  The loud breaths in \
    #   his speech are tough to listed to after a while.",
]

V10_AUDIO_QUALITY_HARSHNESS = [
    # NOTE: Harshness pertaining to buzz, fuzz, hiss, artifacts
    # NOTE: Commented out the following test cases because Alison did not find them noticeable \
    # in datasets after investigating. Test cases are here just to keep a record, in case we want \
    # to revisit them in the future.
    # 10/2022: Jeremy
    # "The Jeremy avatar has an issue where there's a static/electrical noise when he pauses \
    #   between words.",
    # 07/2021: Alana (Alison: Not noticeable in datasets, or model output)
    # "Alana B - sounded great, but then others seemed to have odd artifacts or hiss sounds"
    # 07/2021: Sofia (Alison: Not a problem that needs fixing)
    # "if memory serves, Sofia H had some rather pronounced sibilance"
    # 07/2021: Isabel
    "and then Isabel V had a static-like sound going on",
]

V10_AUDIO_QUALITY_LOUDNESS = [
    # NOTE: 2020-2022
    # NOTE: WSL voices suffer from inconsistent loudness between VA, styles, clips, and within clips
    # Isabel: Too soft
    "My favourite voice was always Isabel V. But, she is too quiet.",
    # Terra: Too loud at the beginning of sentences
    "Terra G. is the avatar. Seems like the beginning of alot of her sentences are loud.",
    # James B.(UK):
    "Whenever I create a voice with James, initially the volume of the audio is fine, but after \
      15-16 seconds of the audio playing, the sound volume goes down automatically.",
    # Wade: Too soft
    "I have noticed that some voices, like Wade, come into Storyline with super low volume.",
    # Wade: Inconsistent loudness between styles
    "Wade's styles have inconsistent loudness. For example, we hear this when switching from his\
      narration to conversational style.",
    # Ava: Too soft
    "Ava M is dramatically lower than Wade C in volume level. The recording quality varies greatly\
      as well.",
]


GREEK_SYMBOLS = [
    # Greek letters for math
    "Α α Β β Γ γ Δ δ Ε ε Ζ ζ Η η Θ θ Ι ι  Κ κ Λ λ Μ μ Ν ν Ξ ξ Ο ο Π π Ρ ρ Σ σ ς Τ τ Υ υ Φ φ Χ χ Ψ \
      ψ Ω ω",
]

ABBREVIATIONS_WITH_VOWELS = [
    # NOTE: These various abbreviations consistenly were mispronounced in v11 on March 1st, 2023.
    "ABBA (musical group) - Agnetha, Björn, Benny, Anni-Frid (first names of the band’s members)",
    "AFK - Away From Keyboard",
    "AFL – American Football League",
    "AGI - Artificial General Intelligence",
    "AWOL - Absent WithOut Leave",
    "CSI - Crime Scene Investigation",
    "DIY - Do It Yourself",
    "EOW - End of Week",
    "FAQ - Frequently Asked Questions",
    "PAWS – Progressive Animal Welfare Society",
    "POTUS - President of the United States",
    "POW - Prisoner Of War",
    "SCOTUS - Supreme Court of the United States",
    "TBA - To Be Announced",
    "TTYL - Talk To You Later",
    "WTH - What The Heck (or Hell)",
    "WWE – World Wrestling Entertainment",
    "YAHOO (search engine) - Yet Another Hierarchical Officious Oracle",
    "YOLO - You Only Live Once",
    "ZIP code - Zone Improvement Plan code",
]

items = locals().items()
TEST_CASES = {k: v for k, v in items if isinstance(v, list) and all(isinstance(t, str) for t in v)}


def generate_test_cases(
    spec_export: SpectrogramModel, test_cases: typing.List[str], seed: int = 123
):
    with fork_rng(seed):
        vocab = sorted(list(spec_export.session_embed.get_vocab()))
        for case in test_cases:
            sesh = random.choice(vocab)
            st.info(f"Seshion: {sesh}\n\nScript: {case}")
            yield griffin_lim_tts(spec_export, XMLType(case), sesh)


Generator = typing.Callable[[SpectrogramModel], typing.Generator[np.ndarray, None, None]]
OPTIONS: typing.Dict[str, Generator]
OPTIONS = {k: partial(generate_test_cases, test_cases=v) for k, v in TEST_CASES.items()}

# TODO: Create a notebook where we go through all the speakers, on various tempos/loudnesses, and
# then review the accuracy.


def generate_annos(
    spec_export: SpectrogramModel,
    anno: typing.Tuple[str, typing.Sequence],
    speakers: typing.Sequence[Speaker] = [
        wsl.TRISTAN_F,  # NOTE: They have a low annotation range around ±2db & ±25%.
        wsl.GIA_V,  # NOTE: They have a low annotation range around ±2db & ±25%.
        wsl.DIARMID_C,  # NOTE: They have a high annotation range around ±4db & ±45%.
        wsl.GARRY_J__STORY,  # NOTE: They have a high annotation range around ±4db & ±45%.
        # NOTE: They have an average range around ±3db & ±30% and have been difficult to work with.
        wsl.JUDE_D__EN_GB,
    ],
):
    for speaker in speakers:
        sesh_vocab = spec_export.session_embed.get_vocab()
        sesh = random.choice([s for s in sesh_vocab if s.spkr == speaker])
        st.info(f"Session: {sesh}")
        tag, range = anno
        for val in range:
            if tag == "tempo":
                val = sesh.spkr_tempo + val
            xml = XMLType(f"<{tag} value='{val}'>{DEFAULT_SCRIPT}</{tag}>")
            wave = griffin_lim_tts(spec_export, xml, sesh)
            audio_len = cf.partial(lib.audio.sample_to_sec)(wave.shape[0])
            tempo = cf.partial(_get_tempo_annotation)(DEFAULT_SCRIPT, audio_len)
            loudness = cf.partial(_get_loudness_annotation)(wave)
            st.info(
                f"- Tag: {tag}={val}\n"
                f"- Tempo: {tempo}\n"
                f"- Generated Griffin-Lim Loudness: {loudness}\n"
            )
            # TODO: Add a loundess computed via spectrogram.
            # TODO: Use a some signal model, and then measure the loudness based on that.
            yield wave


OPTIONS = {
    "LOUDNESS": partial(generate_annos, anno=("loudness", list(lib.utils.arange(6, -7, -3)))),
    "TEMPO": partial(generate_annos, anno=("tempo", list(lib.utils.arange(-0.5, 0.6, 0.25)))),
    **OPTIONS,
}


def main():
    st.markdown("# Test Case Audio Generator")
    st.markdown("Use this workbook to generate batches of audio for evaluating our test cases.")
    run._config.configure(overwrite=True)

    form: DeltaGenerator = st.form(key="form")

    label = "Spectrogram Checkpoints"
    spec_path = st_select_path(label, SPECTROGRAM_MODEL_EXPERIMENTS_PATH, PT_EXTENSION, form)
    items = OPTIONS.items()
    format_test_case_name = lambda i: i[0].replace("_", " ").title()
    option = form.selectbox("Test Cases", items, format_func=format_test_case_name)
    assert option is not None

    if not form.form_submit_button("Submit"):
        return

    spec_ckpt = typing.cast(run.train.spectrogram_model._worker.Checkpoint, load(spec_path))
    spec_export = spec_ckpt.export()

    with st.spinner("Generating audio..."):
        paths = []
        for wave in option[1](spec_export):
            paths.append(audio_to_web_path(wave))
            st_html(f'<audio controls src="{web_path_to_url(paths[-1])}"></audio>')

    with st.spinner("Making Zipfile..."):
        st.text("")
        st_download_files("Audios.zip", "📁 Download Audio(s) (zip)", paths)

    st.success(f"Finished! {lib.utils.mazel_tov()}")


if __name__ == "__main__":
    main()

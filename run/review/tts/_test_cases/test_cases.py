"""Script to hold test cases for tts models"""
import string

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
    # NOTE This section will be reused for testing v11 & all examples here are copied in intialisms section above
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

ALPHABET = [
    # NOTE: Long strings of repetitive letters, can highlight model instability.
    string.ascii_uppercase * 15,
    " ".join(list(string.ascii_uppercase) * 15),
    ". ".join(list(string.ascii_uppercase) * 15),
]

DIFFERENT_SPEAKER = [
    # NOTE: For short single words, the model sometimes also switches speakers, sometimes.
    "Yeah",
    "Lol",
    # NOTE: Big annotaton values, can also cause the model to switch speakers, sometimes.
    '<tempo value="0.25">Please do not do that.</tempo>',
    '<tempo value="5">Please do not do that.</tempo>',
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

SLOW_SCRIPTS = [
    (
        '<tempo value="0.3">Mirabeau B. Lamar was born in Georgia in 1798'
        "</tempo>. He was the son of a wealthy plantation owner. When he "
        'was grown, <tempo value="0.3">Lamar entered Georgia politics</tempo> and served on the '
        'state senate. After <tempo value="0.3">visiting a friend in Texas in 1834</tempo>, he '
        "decided he wanted to live there."
    ),
    (
        '<tempo value="0.4">98 percent of businesses</tempo> estimate that a <tempo value="0.4"> '
        'single hour</tempo> of downtime costs them more than <tempo value="0.4">100,000 dollars'
        "</tempo>."
    ),
    (
        'Ray decided to build a <tempo value="0.7">similar device for his own home</tempo>, and '
        'when he tested it, he was not only did his power spikes go away, but <tempo value="0.6">'
        "his power bill was almost cut in HALF!</tempo>"
    ),
    (
        'Whether it\'s puppy classes <tempo value="0.4"> or playdates with '
        "other dogs</tempo>, exposing your Husky to new experiences can help them to become a "
        '<tempo value="0.7">well-adjusted and friendly companion</tempo>.'
    ),
    (
        'How comfortable will you be in applying what you have learned about <tempo value="0.5">'
        "hydraulic design considerations</tempo> to your next bridge project? Drag the slider from "
        '0 to 10, with <tempo value="0.5"> 10 being the most comfortable</tempo>.'
    ),
    (
        'Are you <tempo value="0.7">drowning in tax burden and don\'t know '
        "what to do?</tempo> Well, bankruptcy might be an option for you, "
        'but it\'s important to understand that <tempo value="0.7">not all taxes can be eliminated '
        "through bankruptcy</tempo>."
    ),
    (
        "And one of the most impactful ways we help you connect with fans is through our "
        '<tempo value="0.6">powerful community of subscribers</tempo>. In fact, we recently '
        '<tempo value="0.7">surpassed 200 million</tempo>. That’s about <tempo value="0.6">3 '
        "percent of the world’s population</tempo> paying monthly so they can have uninterrupted "
        'access to your art. <tempo value="0.3">This is huge</tempo>.'
    ),
    (
        "As you review the details of the scenario, consider the following:\n\nKnowledge of "
        '<tempo value="0.8">current plans, capabilities, and insights</tempo>.\n\nMultiple <tempo '
        'value="0.8">options, actions, and solutions</tempo> for discussion. \n\nNow, let’s see '
        "how this all unfolds and how you and your organization would respond."
    ),
    (
        'Hey this is Major Simpson, the duty officer here at the <tempo value="0.5">six eighteenth '
        "A-O-C, for mission-number alpha-three-seven-five-zero</tempo>. Tell the crew that they’re "
        'done for the day, and we’re going to set them up for a <tempo value="0.5">'
        "two-two-three-zero Zulu</tempo> departure tomorrow.  Have the crew call me A-S-A-P please."
    ),
    (
        "When you are assigned an assessment assessment, take a moment to look at the content "
        'covered in the assessment.  Ask yourself: \n•        Are there <tempo value="0.5">any '
        "resources I will need</tempo> to complete this assessment?\n•        Are there "
        '<tempo value="0.5">any modules I can review</tempo> about this material?\n•        Would '
        'it be beneficial to <tempo value="0.5">read through work instructions or reference guides'
        "</tempo> on this content?"
    ),
    (
        'Organisations should ensure that the tools and equipment are <tempo value="0.6">'
        "thoroughly checked for cuts, cracks, and damages on cords and wires</tempo>. Replacing or "
        'repairing the damaged equipment should be done <tempo value="0.3">immediately</tempo>. '
        'If there is exposure to damaged tools, it is <tempo value="0.4">extremely dangerous'
        "</tempo>."
    ),
    (
        '<tempo value="0.8">Stand tall</tempo> and open your arms <tempo value="0.8">straight to '
        "the side</tempo>. Reach through your fingers and straighten your elbows "
        '<tempo value="0.8">as best you can</tempo>. Slowly start making circles at about the size '
        'of a coconut. Keep your <tempo value="0.8">arms back</tempo> in line with the shoulders '
        'and <tempo value="0.8">feel the nice activation</tempo>.'
    ),
    (
        '<tempo value="0.7">Blood heater coolers</tempo> are used to <tempo value="0.7">maintain '
        "thermo-regulation for patients on ecmo</tempo>. Heat exchange can occur via an "
        '<tempo value="0.7">oxygenator with an integrated heater</tempo> or via a '
        '<tempo value="0.7">stand-alone heat exchanger</tempo> integrated into the circuit. The '
        "heater fluid phase and the blood phase are never in direct contact."
    ),
    (
        'As found in our current <tempo value="0.7">adult brain-dead donor clinical practice '
        'guidelines</tempo>, the goal of fluid management is to <tempo value="0.7">maintain '
        "urinary output greater than or equal to 0.5 milliliters per kilogram per hour</tempo> and "
        'a <tempo value="0.7">CVP 4-12 millimeters of mercury</tempo>. This is in alignment with '
        'the <tempo value="0.7">DMG bundle</tempo>.'
    ),
    (
        'To provide <tempo value="0.5">precise and effective</tempo> care for various eye '
        'conditions, including <tempo value="0.5">hemmorrhagic conditions</tempo>, optometrists '
        "employ a variety of state-of-the-art diagnostic tools and methods, including "
        '<tempo value="0.5">Ocular Ultrasound, Fluorescein Angiography, and Optical Coherence '
        "Tomography</tempo>."
    ),
]

items = locals().items()
TEST_CASES = {k: v for k, v in items if isinstance(v, list) and all(isinstance(t, str) for t in v)}

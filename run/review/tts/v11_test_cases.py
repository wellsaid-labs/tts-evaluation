"""Testing scripts crafted to evaluate TTS Model performance of single words, slow tempo, long
text, and various annotations."""

from long_scripts import LONG_SCRIPTS

SINGLE_WORDS = [
    "augmented",
    "stimulant",
    "query",
    "adventurous",
    "paraplegic",
    "irregularity",
    "unique",
    "mechanism",
    "litmus",
    "mountaintop",
    "Alzheimer's",
    "debilitating",
    "cedar",
    "indicator",
    "calcium",
    "mildew",
    "subrogation",
    "eczema",
    "craft",
    "anemia",
    "enroll",
    "predetermined",
    "whiskey",
    "molasses",
    "lightweight",
    "inscriptions",
    "thoughts",
    "literacy",
    "knowledge",
    "enabled",
    "aperture",
    "unsatisfactory",
    "automatically",
    "Congratulations!",
    "management",
    "legislation",
    "capacity",
    "engage",
    "efficiency",
    "criteria",
    "colleagues",
    "electrical",
    "accessed",
    "protocol",
    "sequential",
    "simultaneously",
    "addresses",
    "buyers",
    "mission",
    "relaxation",
    "proceed",
    "next",
    "Alright",
    "ignored",
    "disconnected",
    "Yes.",
    "Meeting?",
    "example",
    "Goodbye!",
    "no",
    "eight",
    "Seventeen",
    "welcome",
    "then",
    "finally",
    "first",
    "DNA",
    "cobra",
    "rabbit",
    "parrot",
    "ostrich",
    "penguin.",
    "rooster.",
    "elephant.",
    "cat",
    "chicken",
    "crab",
    "goat",
    "squid",
    "antennae",
    "eyes",
    "wings",
    "flying",
    "protection",
    "walking",
    "ECCN",
    "IANA",
    "ACLA",
    "POTUS",
]

AUDIO_CUTOFF = [
    "Taking sides early - I feel like that's a recipe for disaster. It is.",
    "Let me speak to your manager. Egan",
    "I can't believe he then walked away without taking any questions. Wow,",
    "Thanks! For..",
    "using your ears. Why?",
    "Yes. Are you ready to play? Yeah.",
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
V11_TEST_CASES = {
    k: v for k, v in items if isinstance(v, list) and all(isinstance(t, str) for t in v)
}

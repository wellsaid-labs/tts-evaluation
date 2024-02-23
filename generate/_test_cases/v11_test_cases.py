"""Testing scripts crafted to evaluate TTS Model performance of challenging text to establish baseline
and measure progress."""

"""
These clips are here to test if the v11.1 model update performs speaker switching on single words --
a known problem for our pre-beta v11 model. When a single word was rendered, v11 was prone to rendering
that audio in a random voice, so that the clip was very rarely spoken by the selected speaker. This poses
a potential ethics concern: revealing the voices of our custom voices that are not available for public use.
"""
SPEAKER_SWITCH_SINGLE_WORDS_A1a = [
    "Incorporate.",
    "Sterile,",
    "cover",
    '"Signage."',
    "Antihistamines,",
    "Giving.",
    "table",
    "Juxtaposition,",
    "Responsible,",
    "decision",
    "Authority.",
    "Quarantine.",
    "<respell value='HYST'>heist</respell>",
    "Element,",
    "feisty",
    "Application.",
    "<respell value='AY-gyoo'>ague</respell>",
    '"Critical."',
    "<respell value='WURLD'>world</respell>",
    "Fully.",
    "workers",
    "<respell value='EH-furts'>efforts</respell>",
    '"Relations."',
    "Something,",
    "plants",
    "ACBD",
    "JFK",
    "RGB",
    "3D",
    "Y2K",
]

"""
These clips are here to test if the v11.1 model update performs speaker switching on text with max cue values
-- a known problem for our pre-beta v11 model. See note for A1a.
Tempo min max values that elicit speech: 0.4, 3.5
Loudness min max values that elicit speech: -50, 25
"""
SPEAKER_SWITCH_BIG_CUES_A1b = [
    (
        'Let\'s examine <tempo value="0.4"><loudness value="-50">what can affect'
        " establishing a positive</loudness></tempo> safety culture."
    ),
    (
        'If you\’ve ever cyberbullied someone, <tempo value="0.4"><loudness value="25">'
        "you may have thought it was funny</loudness></tempo> at the time."
    ),
    (
        'It provides <tempo value="3.5"><loudness value="25">exceptional adhesion '
        "on bare metal</loudness></tempo>."
    ),
    (
        'Thanks for asking. <tempo value="3.5"><loudness value="-50">I haven\'t been able '
        "to find the right links</loudness></tempo>. Can you help me?"
    ),
    (
        '<loudness value="17">Looking for a Strategic Partner</loudness> for High-Performance'
        " software solutions?"
    ),
    (
        '<loudness value="-20">The Identification Documents page</loudness> displays your'
        " identification documents."
    ),
    ('a report is <tempo value="0.4">also</tempo> generated.'),
    ('<tempo value="3.5">Considerations</tempo> When Hosting a Web Meeting'),
]

"""
These clips are here to test the v11 model's performance on a known error for v10: audio cutoff.
When a clip ended in a sentence or beginning clause that consisted of 4 characters or less, the
model cut off that final sentence or sentence fragment. Of these test cases, all but the first
experience this error in our current v10 model in prod.
"""

AUDIO_CUTOFF = [
    "Taking sides early - I feel like that's a recipe for disaster. It is.",
    "Let me speak to your manager. Egan",
    "I can't believe he then walked away without taking any questions. Wow,",
    "Thanks! For..",
    "using your ears. Why?",
    "Yes. Are you ready to play? Yeah.",
]

"""
These scripts were created using user text, with an assortment of phrases slowed down to evaluate
the potential impact of reduced tempo on pronunciation.
"""
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

LOUDNESS_MIN__WORD = [
    'a report is <loudness value="-20">also</loudness> generated.',
    '<loudness value="-20">Click</loudness> on clean piercing needle.',
    'Social Media <loudness value="-20">helps</loudness> brands...',
    'and the Box will pop up <loudness value="-20">Below</loudness>',
    'Focus on benefits, <loudness value="-20">not</loudness> just features: .',
    '<loudness value="-20">Considerations</loudness> When Hosting a Web Meeting',
    'And diets and exercise <loudness value="-20">add</loudness> to the struggle.',
    'To all women <loudness value="-20">struggling</loudness> with puffy under eyes',
    'and I\'ll get back to you as <loudness value="-20">soon</loudness> as possible.',
    'determining the <loudness value="-20">most</loudness> effective burn treatments.',
    'And what <loudness value="-20">about</loudness> sending international transfers?',
    'Which led him to a <loudness value="-20">deeply</loudness> disturbing conclusion.',
    'There are <loudness value="-20">two</loudness> ways to create Clauses in I-see-I.',
    'Decrease <loudness value="-20">significant</loudness> financial and labor expenses.',
    '2. Help aids in <loudness value="-20">reducing</loudness> soreness and bruising',
]

LOUDNESS_MAX__WORD = [
    'a report is <loudness value="17">also</loudness> generated.',
    '<loudness value="17">Click</loudness> on clean piercing needle.',
    'Social Media <loudness value="17">helps</loudness> brands...',
    'and the Box will pop up <loudness value="17">Below</loudness>',
    'Focus on benefits, <loudness value="17">not</loudness> just features: .',
    '<loudness value="17">Considerations</loudness> When Hosting a Web Meeting',
    'And diets and exercise <loudness value="17">add</loudness> to the struggle.',
    'To all women <loudness value="17">struggling</loudness> with puffy under eyes',
    'and I\'ll get back to you as <loudness value="17">soon</loudness> as possible.',
    'determining the <loudness value="17">most</loudness> effective burn treatments.',
    'And what <loudness value="17">about</loudness> sending international transfers?',
    'Which led him to a <loudness value="17">deeply</loudness> disturbing conclusion.',
    'There are <loudness value="17">two</loudness> ways to create Clauses in I-see-I.',
    'Decrease <loudness value="17">significant</loudness> financial and labor expenses.',
    '2. Help aids in <loudness value="17">reducing</loudness> soreness and bruising',
]

LOUDNESS_MIN__CLAUSE = [
    (
        '<loudness value="-20">The Identification Documents page</loudness> displays your'
        " identification documents."
    ),
    (
        '<loudness value="-20">This chiropractor designed</loudness> the device using breakthrough'
        " NMES technology."
    ),
    (
        'Depending on your team, <loudness value="-20">you may also need to update</loudness> the'
        " Meeting Type."
    ),
    (
        'Everyday, and every way, <loudness value="-20">i am becoming,</loudness> a greater'
        " version, of myself."
    ),
    (
        'Let\'s examine <loudness value="-20">what can affect</loudness> establishing a positive'
        " safety culture."
    ),
    (
        'Employee Templates help save time to <loudness value="-20">streamline employee'
        " setup</loudness> in Meevo."
    ),
    (
        '<loudness value="-20">Looking for a Strategic Partner</loudness> for High-Performance'
        " software solutions?"
    ),
    (
        'Next, you <loudness value="-20">use Category filter "Ratings"</loudness> to find the "M.'
        ' L. F. I." field.'
    ),
    (
        'The person highest in the pecking order tends to <loudness value="-20">tell others what to'
        " do.</loudness>"
    ),
    (
        '<loudness value="-20">We have also included</loudness> links to both documents below for'
        " your convenience"
    ),
    (
        '<loudness value="-20">The Information section lists</loudness> the Title, URL Name, and'
        " Article Type."
    ),
    (
        'Click the Secret asterix, here <loudness value="-20">you will enter</loudness> your FOD'
        " Tenant name..."
    ),
    (
        '<loudness value="-20">This trick to get gas for a penny is going to get banned</loudness>'
        " in Canada."
    ),
    (
        '<loudness value="-20">Segregate the following items</loudness> into waste that require'
        " special handling."
    ),
    (
        '<loudness value="-20">The PTO Requests page allows</loudness> you to manage your paid time'
        " off requests."
    ),
]

LOUDNESS_MAX__CLAUSE = [
    (
        '<loudness value="17">The Identification Documents page</loudness> displays your'
        " identification documents."
    ),
    (
        '<loudness value="17">This chiropractor designed</loudness> the device using breakthrough'
        " NMES technology."
    ),
    (
        'Depending on your team, <loudness value="17">you may also need to update</loudness> the'
        " Meeting Type."
    ),
    (
        'Everyday, and every way, <loudness value="17">i am becoming,</loudness> a greater version,'
        " of myself."
    ),
    (
        'Let\'s examine <loudness value="17">what can affect</loudness> establishing a positive'
        " safety culture."
    ),
    (
        'Employee Templates help save time to <loudness value="17">streamline employee'
        " setup</loudness> in Meevo."
    ),
    (
        '<loudness value="17">Looking for a Strategic Partner</loudness> for High-Performance'
        " software solutions?"
    ),
    (
        'Next, you <loudness value="17">use Category filter "Ratings"</loudness> to find the "M. L.'
        ' F. I." field.'
    ),
    (
        'The person highest in the pecking order tends to <loudness value="17">tell others what to'
        " do.</loudness>"
    ),
    (
        '<loudness value="17">We have also included</loudness> links to both documents below for'
        " your convenience"
    ),
    (
        '<loudness value="17">The Information section lists</loudness> the Title, URL Name, and'
        " Article Type."
    ),
    (
        'Click the Secret asterix, here <loudness value="17">you will enter</loudness> your FOD'
        " Tenant name..."
    ),
    (
        '<loudness value="17">This trick to get gas for a penny is going to get banned</loudness>'
        " in Canada."
    ),
    (
        '<loudness value="17">Segregate the following items</loudness> into waste that require'
        " special handling."
    ),
    (
        '<loudness value="17">The PTO Requests page allows</loudness> you to manage your paid time'
        " off requests."
    ),
]

LOUDNESS_MIN__SENTENCE = [
    (
        '<loudness value="-20">Every product has gone through multiple on the job tests to refine'
        " it and guarantee it meets the tough standards of the HVACR industry.</loudness> The same"
        " quality and service you’ve come to know from Habegger over the years is now available in"
        " a variety of hvac chemicals. From coil cleaners and lineset flush to leak sealants and"
        " vacuum pump oil, Hab kim is now offering chemicals your customers can depend on for any"
        " job they need to tackle."
    ),
    (
        "Finally, we need to mention some additional dimensions available in the system, including"
        ' Restatement, Disposal and Currency Adjustment. <loudness value="-20">Data in these'
        " dimensions are loaded mainly to the defaul value, None, with some exceptions.</loudness>"
        " For instance, the adjustments performed by the consolidation team in HFM can be posted to"
        " dollar or euro depending on the currency of the adjustment."
    ),
    (
        "I asked his username and it was him. 2000 miles away from where I first virtually met this"
        ' guy he’s sitting next to me in math class. <loudness value="-20">When I told him mine his'
        " face went into total shock and he didn’t know how to respond.</loudness> It was awkward"
        " for sure."
    ),
    (
        "Several factors can influence systolic blood pressure, including age, gender, body mass"
        ' index, and physical activity levels. <loudness value="-20">Lifestyle choices, such as'
        " consuming excessive salt and alcohol, smoking, and experiencing stress, can also"
        " contribute to elevated systolic blood pressure.</loudness> Diastolic blood pressure, the"
        " lower of the two readings, refers to the pressure in the arteries between heart beats."
    ),
    (
        "As a Jeminigh, you are known for your adaptability and social nature. <loudness"
        ' value="-20">However, it\'s important to know when to draw the line and take care of'
        " yourself.</loudness> Establishing healthy boundaries will allow you to prioritize your"
        " own needs and prevent burnout."
    ),
    (
        "Of course, forgiving your spouse doesn't entail being a coward or tolerating abuse. It's"
        " important to be conscious of the many non-physical signs of an abusive relationship."
        ' <loudness value="-20">You must first decide to accept them fully in order to have a'
        " successful relationship.</loudness>"
    ),
    (
        '<loudness value="-20">While there is some support for Kohlberg\'s theory, there are also'
        " limitations and criticisms of the theory.</loudness> Longitudinal studies have found that"
        " individuals generally progress through the stages of moral development in a sequential"
        " order, without skipping stages. Furthermore, there is evidence to suggest that moral"
        " reasoning and moral behavior are linked, as individuals who reason at higher levels are"
        " more likely to behave morally."
    ),
    (
        "Sparklines are mini charts that can be inserted inside the cells. <loudness"
        ' value="-20">Sparklines are great for showing trends in a series of values, such as'
        " seasonal increases or decreases and economic cycles.</loudness> They also make you able"
        " to highlight maximum and minimum values."
    ),
    (
        'From the main menu, expand the data views and reports. <loudness value="-20">Select report'
        " library, click run report, expand custom reports, click on store weekly schedule or daily"
        " break schedule and click select.</loudness> Select the timeframe, location, Output, and"
        " click run report."
    ),
    (
        "This situation could benefit from a relief structure through the embankment on the"
        ' relatively wide right floodplain. <loudness value="-20">Therefore, another alternative'
        " could be combining the original 400-foot bridge with a 150-foot relief bridge.</loudness>"
        " The two alternatives provide 150 feet of additional bridge length compared to the"
        " 400-foot bridge, so we can compare the hydraulic performance of these alternatives."
    ),
    (
        "Machine learning uses an algorithm to analyze data, learn from it, and make decisions"
        ' based on what it learned. <loudness value="-20">Deep learning is a subset of machine'
        " learning.</loudness> In deep learning, the algorithm can actually learn as a brain learns"
        " through the artificial neural networks."
    ),
    (
        "Each client is paired with a dedicated team member who will be your guide from start to"
        " finish, which includes a project manager, a design & sourcing expert, a master"
        ' patternmaker, and a seasoned seamstress. <loudness value="-20">The Visualz Group prides'
        " itself on training the future of fashion and is proud to have helped launch the careers"
        " for hundreds of young designers.</loudness> No question is too big or too small."
    ),
    (
        '<loudness value="-20">Today, we\'re going to talk about one of the most underrated forms'
        " of exercise out there: running.</loudness> It's an activity that is often overlooked or"
        " dismissed as too difficult, too time-consuming, or just plain boring. But the truth is,"
        " running is one of the most beneficial things you can do for your body and mind, and it's"
        " accessible to almost everyone."
    ),
    (
        '<loudness value="-20">tendoneyetis causes aching or stabbing pain, tenderness, and'
        " stiffness.</loudness> tendonitis is usually associated with a particular body part; for"
        " example, Achilles tendonitis or patellar tendonitis. Early treatment usually starts with"
        " ice, rest, and anti-inflammatories and can prevent chronic problems that require surgery."
    ),
    (
        '<loudness value="-20">The famous statistician Edwards Deming once said, "In God we Trust:'
        ' All others must bring Data ".</loudness> That’s how important "Data" is. A regularly'
        " scheduled Data Analysis not just gives us an overview of our tasks but also allows us to"
        " detect systematic errors, as well as to evaluate and prevent the associated risks."
    ),
]

LOUDNESS_MAX__SENTENCE = [
    (
        '<loudness value="17">Every product has gone through multiple on the job tests to refine it'
        " and guarantee it meets the tough standards of the HVACR industry.</loudness> The same"
        " quality and service you’ve come to know from Habegger over the years is now available in"
        " a variety of hvac chemicals. From coil cleaners and lineset flush to leak sealants and"
        " vacuum pump oil, Hab kim is now offering chemicals your customers can depend on for any"
        " job they need to tackle."
    ),
    (
        "Finally, we need to mention some additional dimensions available in the system, including"
        ' Restatement, Disposal and Currency Adjustment. <loudness value="17">Data in these'
        " dimensions are loaded mainly to the defaul value, None, with some exceptions.</loudness>"
        " For instance, the adjustments performed by the consolidation team in HFM can be posted to"
        " dollar or euro depending on the currency of the adjustment."
    ),
    (
        "I asked his username and it was him. 2000 miles away from where I first virtually met this"
        ' guy he’s sitting next to me in math class. <loudness value="17">When I told him mine his'
        " face went into total shock and he didn’t know how to respond.</loudness> It was awkward"
        " for sure."
    ),
    (
        "Several factors can influence systolic blood pressure, including age, gender, body mass"
        ' index, and physical activity levels. <loudness value="17">Lifestyle choices, such as'
        " consuming excessive salt and alcohol, smoking, and experiencing stress, can also"
        " contribute to elevated systolic blood pressure.</loudness> Diastolic blood pressure, the"
        " lower of the two readings, refers to the pressure in the arteries between heart beats."
    ),
    (
        "As a Jeminigh, you are known for your adaptability and social nature. <loudness"
        ' value="17">However, it\'s important to know when to draw the line and take care of'
        " yourself.</loudness> Establishing healthy boundaries will allow you to prioritize your"
        " own needs and prevent burnout."
    ),
    (
        "Of course, forgiving your spouse doesn't entail being a coward or tolerating abuse. It's"
        " important to be conscious of the many non-physical signs of an abusive relationship."
        ' <loudness value="17">You must first decide to accept them fully in order to have a'
        " successful relationship.</loudness>"
    ),
    (
        '<loudness value="17">While there is some support for Kohlberg\'s theory, there are also'
        " limitations and criticisms of the theory.</loudness> Longitudinal studies have found that"
        " individuals generally progress through the stages of moral development in a sequential"
        " order, without skipping stages. Furthermore, there is evidence to suggest that moral"
        " reasoning and moral behavior are linked, as individuals who reason at higher levels are"
        " more likely to behave morally."
    ),
    (
        "Sparklines are mini charts that can be inserted inside the cells. <loudness"
        ' value="17">Sparklines are great for showing trends in a series of values, such as'
        " seasonal increases or decreases and economic cycles.</loudness> They also make you able"
        " to highlight maximum and minimum values."
    ),
    (
        'From the main menu, expand the data views and reports. <loudness value="17">Select report'
        " library, click run report, expand custom reports, click on store weekly schedule or daily"
        " break schedule and click select.</loudness> Select the timeframe, location, Output, and"
        " click run report."
    ),
    (
        "This situation could benefit from a relief structure through the embankment on the"
        ' relatively wide right floodplain. <loudness value="17">Therefore, another alternative'
        " could be combining the original 400-foot bridge with a 150-foot relief bridge.</loudness>"
        " The two alternatives provide 150 feet of additional bridge length compared to the"
        " 400-foot bridge, so we can compare the hydraulic performance of these alternatives."
    ),
    (
        "Machine learning uses an algorithm to analyze data, learn from it, and make decisions"
        ' based on what it learned. <loudness value="17">Deep learning is a subset of machine'
        " learning.</loudness> In deep learning, the algorithm can actually learn as a brain learns"
        " through the artificial neural networks."
    ),
    (
        "Each client is paired with a dedicated team member who will be your guide from start to"
        " finish, which includes a project manager, a design & sourcing expert, a master"
        ' patternmaker, and a seasoned seamstress. <loudness value="17">The Visualz Group prides'
        " itself on training the future of fashion and is proud to have helped launch the careers"
        " for hundreds of young designers.</loudness> No question is too big or too small."
    ),
    (
        '<loudness value="17">Today, we\'re going to talk about one of the most underrated forms'
        " of exercise out there: running.</loudness> It's an activity that is often overlooked or"
        " dismissed as too difficult, too time-consuming, or just plain boring. But the truth is,"
        " running is one of the most beneficial things you can do for your body and mind, and it's"
        " accessible to almost everyone."
    ),
    (
        '<loudness value="17">tendoneyetis causes aching or stabbing pain, tenderness, and'
        " stiffness.</loudness> tendonitis is usually associated with a particular body part; for"
        " example, Achilles tendonitis or patellar tendonitis. Early treatment usually starts with"
        " ice, rest, and anti-inflammatories and can prevent chronic problems that require surgery."
    ),
    (
        '<loudness value="17">The famous statistician Edwards Deming once said, "In God we Trust:'
        ' All others must bring Data ".</loudness> That’s how important "Data" is. A regularly'
        " scheduled Data Analysis not just gives us an overview of our tasks but also allows us to"
        " detect systematic errors, as well as to evaluate and prevent the associated risks."
    ),
]

TEMPO_MIN__WORD = [
    'a report is <tempo value="0.4">also</tempo> generated.',
    '<tempo value="0.4">Click</tempo> on clean piercing needle.',
    'Social Media <tempo value="0.4">helps</tempo> brands...',
    'and the Box will pop up <tempo value="0.4">Below</tempo>',
    'Focus on benefits, <tempo value="0.4">not</tempo> just features: .',
    '<tempo value="0.4">Considerations</tempo> When Hosting a Web Meeting',
    'And diets and exercise <tempo value="0.4">add</tempo> to the struggle.',
    'To all women <tempo value="0.4">struggling</tempo> with puffy under eyes',
    'and I\'ll get back to you as <tempo value="0.4">soon</tempo> as possible.',
    'determining the <tempo value="0.4">most</tempo> effective burn treatments.',
    'And what <tempo value="0.4">about</tempo> sending international transfers?',
    'Which led him to a <tempo value="0.4">deeply</tempo> disturbing conclusion.',
    'There are <tempo value="0.4">two</tempo> ways to create Clauses in I-see-I.',
    'Decrease <tempo value="0.4">significant</tempo> financial and labor expenses.',
    '2. Help aids in <tempo value="0.4">reducing</tempo> soreness and bruising',
]

TEMPO_MAX__WORD = [
    'a report is <tempo value="3.5">also</tempo> generated.',
    '<tempo value="3.5">Click</tempo> on clean piercing needle.',
    'Social Media <tempo value="3.5">helps</tempo> brands...',
    'and the Box will pop up <tempo value="3.5">Below</tempo>',
    'Focus on benefits, <tempo value="3.5">not</tempo> just features: .',
    '<tempo value="3.5">Considerations</tempo> When Hosting a Web Meeting',
    'And diets and exercise <tempo value="3.5">add</tempo> to the struggle.',
    'To all women <tempo value="3.5">struggling</tempo> with puffy under eyes',
    'and I\'ll get back to you as <tempo value="3.5">soon</tempo> as possible.',
    'determining the <tempo value="3.5">most</tempo> effective burn treatments.',
    'And what <tempo value="3.5">about</tempo> sending international transfers?',
    'Which led him to a <tempo value="3.5">deeply</tempo> disturbing conclusion.',
    'There are <tempo value="3.5">two</tempo> ways to create Clauses in I-see-I.',
    'Decrease <tempo value="3.5">significant</tempo> financial and labor expenses.',
    '2. Help aids in <tempo value="3.5">reducing</tempo> soreness and bruising',
]

TEMPO_MIN__CLAUSE = [
    (
        '<tempo value="0.4">The Identification Documents page</tempo> displays your identification'
        " documents."
    ),
    (
        '<tempo value="0.4">This chiropractor designed</tempo> the device using breakthrough NMES'
        " technology."
    ),
    (
        'Depending on your team, <tempo value="0.4">you may also need to update</tempo> the Meeting'
        " Type."
    ),
    (
        'Everyday, and every way, <tempo value="0.4">i am becoming,</tempo> a greater version, of'
        " myself."
    ),
    (
        'Let\'s examine <tempo value="0.4">what can affect</tempo> establishing a positive safety'
        " culture."
    ),
    (
        'Employee Templates help save time to <tempo value="0.4">streamline employee setup</tempo>'
        " in Meevo."
    ),
    (
        '<tempo value="0.4">Looking for a Strategic Partner</tempo> for High-Performance software'
        " solutions?"
    ),
    (
        'Next, you <tempo value="0.4">use Category filter "Ratings"</tempo> to find the "M. L. F.'
        ' I." field.'
    ),
    (
        'The person highest in the pecking order tends to <tempo value="0.4">tell others what to'
        " do.</tempo>"
    ),
    (
        '<tempo value="0.4">We have also included</tempo> links to both documents below for your'
        " convenience"
    ),
    (
        '<tempo value="0.4">The Information section lists</tempo> the Title, URL Name, and Article'
        " Type."
    ),
    (
        'Click the Secret asterix, here <tempo value="0.4">you will enter</tempo> your FOD Tenant'
        " name..."
    ),
    (
        '<tempo value="0.4">This trick to get gas for a penny is going to get banned</tempo> in'
        " Canada."
    ),
    (
        '<tempo value="0.4">Segregate the following items</tempo> into waste that require special'
        " handling."
    ),
    (
        '<tempo value="0.4">The PTO Requests page allows</tempo> you to manage your paid time off'
        " requests."
    ),
]

TEMPO_MAX__CLAUSE = [
    (
        '<tempo value="3.5">The Identification Documents page</tempo> displays your identification'
        " documents."
    ),
    (
        '<tempo value="3.5">This chiropractor designed</tempo> the device using breakthrough NMES'
        " technology."
    ),
    (
        'Depending on your team, <tempo value="3.5">you may also need to update</tempo> the Meeting'
        " Type."
    ),
    (
        'Everyday, and every way, <tempo value="3.5">i am becoming,</tempo> a greater version, of'
        " myself."
    ),
    (
        'Let\'s examine <tempo value="3.5">what can affect</tempo> establishing a positive safety'
        " culture."
    ),
    (
        'Employee Templates help save time to <tempo value="3.5">streamline employee setup</tempo>'
        " in Meevo."
    ),
    (
        '<tempo value="3.5">Looking for a Strategic Partner</tempo> for High-Performance software'
        " solutions?"
    ),
    (
        'Next, you <tempo value="3.5">use Category filter "Ratings"</tempo> to find the "M. L. F.'
        ' I." field.'
    ),
    (
        'The person highest in the pecking order tends to <tempo value="3.5">tell others what to'
        " do.</tempo>"
    ),
    (
        '<tempo value="3.5">We have also included</tempo> links to both documents below for your'
        " convenience"
    ),
    (
        '<tempo value="3.5">The Information section lists</tempo> the Title, URL Name, and Article'
        " Type."
    ),
    (
        'Click the Secret asterix, here <tempo value="3.5">you will enter</tempo> your FOD Tenant'
        " name..."
    ),
    (
        '<tempo value="3.5">This trick to get gas for a penny is going to get banned</tempo> in'
        " Canada."
    ),
    (
        '<tempo value="3.5">Segregate the following items</tempo> into waste that require special'
        " handling."
    ),
    (
        '<tempo value="3.5">The PTO Requests page allows</tempo> you to manage your paid time off'
        " requests."
    ),
]

TEMPO_MIN__SENTENCE = [
    (
        '<tempo value="0.4">Every product has gone through multiple on the job tests to refine it'
        " and guarantee it meets the tough standards of the HVACR industry.</tempo> The same"
        " quality and service you’ve come to know from Habegger over the years is now available in"
        " a variety of hvac chemicals. From coil cleaners and lineset flush to leak sealants and"
        " vacuum pump oil, Hab kim is now offering chemicals your customers can depend on for any"
        " job they need to tackle."
    ),
    (
        "Finally, we need to mention some additional dimensions available in the system, including"
        ' Restatement, Disposal and Currency Adjustment. <tempo value="0.4">Data in these'
        " dimensions are loaded mainly to the defaul value, None, with some exceptions.</tempo> For"
        " instance, the adjustments performed by the consolidation team in HFM can be posted to"
        " dollar or euro depending on the currency of the adjustment."
    ),
    (
        "I asked his username and it was him. 2000 miles away from where I first virtually met this"
        ' guy he’s sitting next to me in math class. <tempo value="0.4">When I told him mine his'
        " face went into total shock and he didn’t know how to respond.</tempo> It was awkward for"
        " sure."
    ),
    (
        "Several factors can influence systolic blood pressure, including age, gender, body mass"
        ' index, and physical activity levels. <tempo value="0.4">Lifestyle choices, such as'
        " consuming excessive salt and alcohol, smoking, and experiencing stress, can also"
        " contribute to elevated systolic blood pressure.</tempo> Diastolic blood pressure, the"
        " lower of the two readings, refers to the pressure in the arteries between heart beats."
    ),
    (
        "As a Jeminigh, you are known for your adaptability and social nature. <tempo"
        ' value="0.4">However, it\'s important to know when to draw the line and take care of'
        " yourself.</tempo> Establishing healthy boundaries will allow you to prioritize your own"
        " needs and prevent burnout."
    ),
    (
        "Of course, forgiving your spouse doesn't entail being a coward or tolerating abuse. It's"
        " important to be conscious of the many non-physical signs of an abusive relationship."
        ' <tempo value="0.4">You must first decide to accept them fully in order to have a'
        " successful relationship.</tempo>"
    ),
    (
        '<tempo value="0.4">While there is some support for Kohlberg\'s theory, there are also'
        " limitations and criticisms of the theory.</tempo> Longitudinal studies have found that"
        " individuals generally progress through the stages of moral development in a sequential"
        " order, without skipping stages. Furthermore, there is evidence to suggest that moral"
        " reasoning and moral behavior are linked, as individuals who reason at higher levels are"
        " more likely to behave morally."
    ),
    (
        "Sparklines are mini charts that can be inserted inside the cells. <tempo"
        ' value="0.4">Sparklines are great for showing trends in a series of values, such as'
        " seasonal increases or decreases and economic cycles.</tempo> They also make you able to"
        " highlight maximum and minimum values."
    ),
    (
        'From the main menu, expand the data views and reports. <tempo value="0.4">Select report'
        " library, click run report, expand custom reports, click on store weekly schedule or daily"
        " break schedule and click select.</tempo> Select the timeframe, location, Output, and"
        " click run report."
    ),
    (
        "This situation could benefit from a relief structure through the embankment on the"
        ' relatively wide right floodplain. <tempo value="0.4">Therefore, another alternative could'
        " be combining the original 400-foot bridge with a 150-foot relief bridge.</tempo> The two"
        " alternatives provide 150 feet of additional bridge length compared to the 400-foot"
        " bridge, so we can compare the hydraulic performance of these alternatives."
    ),
    (
        "Machine learning uses an algorithm to analyze data, learn from it, and make decisions"
        ' based on what it learned. <tempo value="0.4">Deep learning is a subset of machine'
        " learning.</tempo> In deep learning, the algorithm can actually learn as a brain learns"
        " through the artificial neural networks."
    ),
    (
        "Each client is paired with a dedicated team member who will be your guide from start to"
        " finish, which includes a project manager, a design & sourcing expert, a master"
        ' patternmaker, and a seasoned seamstress. <tempo value="0.4">The Visualz Group prides'
        " itself on training the future of fashion and is proud to have helped launch the careers"
        " for hundreds of young designers.</tempo> No question is too big or too small."
    ),
    (
        '<tempo value="0.4">Today, we\'re going to talk about one of the most underrated forms of'
        " exercise out there: running.</tempo> It's an activity that is often overlooked or"
        " dismissed as too difficult, too time-consuming, or just plain boring. But the truth is,"
        " running is one of the most beneficial things you can do for your body and mind, and it's"
        " accessible to almost everyone."
    ),
    (
        '<tempo value="0.4">tendoneyetis causes aching or stabbing pain, tenderness, and'
        " stiffness.</tempo> tendonitis is usually associated with a particular body part; for"
        " example, Achilles tendonitis or patellar tendonitis. Early treatment usually starts with"
        " ice, rest, and anti-inflammatories and can prevent chronic problems that require surgery."
    ),
    (
        '<tempo value="0.4">The famous statistician Edwards Deming once said, "In God we Trust: All'
        ' others must bring Data ".</tempo> That’s how important "Data" is. A regularly scheduled'
        " Data Analysis not just gives us an overview of our tasks but also allows us to detect"
        " systematic errors, as well as to evaluate and prevent the associated risks."
    ),
]

TEMPO_MAX__SENTENCE = [
    (
        '<tempo value="3.5">Every product has gone through multiple on the job tests to refine it'
        " and guarantee it meets the tough standards of the HVACR industry.</tempo> The same"
        " quality and service you’ve come to know from Habegger over the years is now available in"
        " a variety of hvac chemicals. From coil cleaners and lineset flush to leak sealants and"
        " vacuum pump oil, Hab kim is now offering chemicals your customers can depend on for any"
        " job they need to tackle."
    ),
    (
        "Finally, we need to mention some additional dimensions available in the system, including"
        ' Restatement, Disposal and Currency Adjustment. <tempo value="3.5">Data in these'
        " dimensions are loaded mainly to the defaul value, None, with some exceptions.</tempo> For"
        " instance, the adjustments performed by the consolidation team in HFM can be posted to"
        " dollar or euro depending on the currency of the adjustment."
    ),
    (
        "I asked his username and it was him. 2000 miles away from where I first virtually met this"
        ' guy he’s sitting next to me in math class. <tempo value="3.5">When I told him mine his'
        " face went into total shock and he didn’t know how to respond.</tempo> It was awkward for"
        " sure."
    ),
    (
        "Several factors can influence systolic blood pressure, including age, gender, body mass"
        ' index, and physical activity levels. <tempo value="3.5">Lifestyle choices, such as'
        " consuming excessive salt and alcohol, smoking, and experiencing stress, can also"
        " contribute to elevated systolic blood pressure.</tempo> Diastolic blood pressure, the"
        " lower of the two readings, refers to the pressure in the arteries between heart beats."
    ),
    (
        "As a Jeminigh, you are known for your adaptability and social nature. <tempo"
        ' value="3.5">However, it\'s important to know when to draw the line and take care of'
        " yourself.</tempo> Establishing healthy boundaries will allow you to prioritize your own"
        " needs and prevent burnout."
    ),
    (
        "Of course, forgiving your spouse doesn't entail being a coward or tolerating abuse. It's"
        " important to be conscious of the many non-physical signs of an abusive relationship."
        ' <tempo value="3.5">You must first decide to accept them fully in order to have a'
        " successful relationship.</tempo>"
    ),
    (
        '<tempo value="3.5">While there is some support for Kohlberg\'s theory, there are also'
        " limitations and criticisms of the theory.</tempo> Longitudinal studies have found that"
        " individuals generally progress through the stages of moral development in a sequential"
        " order, without skipping stages. Furthermore, there is evidence to suggest that moral"
        " reasoning and moral behavior are linked, as individuals who reason at higher levels are"
        " more likely to behave morally."
    ),
    (
        "Sparklines are mini charts that can be inserted inside the cells. <tempo"
        ' value="3.5">Sparklines are great for showing trends in a series of values, such as'
        " seasonal increases or decreases and economic cycles.</tempo> They also make you able to"
        " highlight maximum and minimum values."
    ),
    (
        'From the main menu, expand the data views and reports. <tempo value="3.5">Select report'
        " library, click run report, expand custom reports, click on store weekly schedule or daily"
        " break schedule and click select.</tempo> Select the timeframe, location, Output, and"
        " click run report."
    ),
    (
        "This situation could benefit from a relief structure through the embankment on the"
        ' relatively wide right floodplain. <tempo value="3.5">Therefore, another alternative could'
        " be combining the original 400-foot bridge with a 150-foot relief bridge.</tempo> The two"
        " alternatives provide 150 feet of additional bridge length compared to the 400-foot"
        " bridge, so we can compare the hydraulic performance of these alternatives."
    ),
    (
        "Machine learning uses an algorithm to analyze data, learn from it, and make decisions"
        ' based on what it learned. <tempo value="3.5">Deep learning is a subset of machine'
        " learning.</tempo> In deep learning, the algorithm can actually learn as a brain learns"
        " through the artificial neural networks."
    ),
    (
        "Each client is paired with a dedicated team member who will be your guide from start to"
        " finish, which includes a project manager, a design & sourcing expert, a master"
        ' patternmaker, and a seasoned seamstress. <tempo value="3.5">The Visualz Group prides'
        " itself on training the future of fashion and is proud to have helped launch the careers"
        " for hundreds of young designers.</tempo> No question is too big or too small."
    ),
    (
        '<tempo value="3.5">Today, we\'re going to talk about one of the most underrated forms of'
        " exercise out there: running.</tempo> It's an activity that is often overlooked or"
        " dismissed as too difficult, too time-consuming, or just plain boring. But the truth is,"
        " running is one of the most beneficial things you can do for your body and mind, and it's"
        " accessible to almost everyone."
    ),
    (
        '<tempo value="3.5">tendoneyetis causes aching or stabbing pain, tenderness, and'
        " stiffness.</tempo> tendonitis is usually associated with a particular body part; for"
        " example, Achilles tendonitis or patellar tendonitis. Early treatment usually starts with"
        " ice, rest, and anti-inflammatories and can prevent chronic problems that require surgery."
    ),
    (
        '<tempo value="3.5">The famous statistician Edwards Deming once said, "In God we Trust: All'
        ' others must bring Data ".</tempo> That’s how important "Data" is. A regularly scheduled'
        " Data Analysis not just gives us an overview of our tasks but also allows us to detect"
        " systematic errors, as well as to evaluate and prevent the associated risks."
    ),
]

TEMPO_MIN__LOUDNESS_MIN__CLAUSE = [
    (
        '<tempo value="0.4"><loudness value="-20">This trick to get gas for a penny is going to get'
        " banned</loudness></tempo> in Canada."
    ),
    (
        'Let\'s examine <tempo value="0.4"><loudness value="-20">what can affect</loudness></tempo>'
        " establishing a positive safety culture."
    ),
    (
        '<tempo value="0.4"><loudness value="-20">This chiropractor designed</loudness></tempo> the'
        " device using breakthrough NMES technology."
    ),
    (
        '<tempo value="0.4"><loudness value="-20">Looking for a Strategic'
        " Partner</loudness></tempo> for High-Performance software solutions?"
    ),
    (
        'Depending on your team, <tempo value="0.4"><loudness value="-20">you may also need to'
        " update</loudness></tempo> the Meeting Type."
    ),
]

TEMPO_MIN__LOUDNESS_MAX__CLAUSE = [
    (
        '<tempo value="0.4"><loudness value="17">This trick to get gas for a penny is going to get'
        " banned</loudness></tempo> in Canada."
    ),
    (
        'Let\'s examine <tempo value="0.4"><loudness value="17">what can affect</loudness></tempo>'
        " establishing a positive safety culture."
    ),
    (
        '<tempo value="0.4"><loudness value="17">This chiropractor designed</loudness></tempo> the'
        " device using breakthrough NMES technology."
    ),
    (
        '<tempo value="0.4"><loudness value="17">Looking for a Strategic Partner</loudness></tempo>'
        " for High-Performance software solutions?"
    ),
    (
        'Depending on your team, <tempo value="0.4"><loudness value="17">you may also need to'
        " update</loudness></tempo> the Meeting Type."
    ),
]

TEMPO_MAX__LOUDNESS_MIN__CLAUSE = [
    (
        '<tempo value="3.5"><loudness value="-20">This trick to get gas for a penny is going to get'
        " banned</loudness></tempo> in Canada."
    ),
    (
        'Let\'s examine <tempo value="3.5"><loudness value="-20">what can affect</loudness></tempo>'
        " establishing a positive safety culture."
    ),
    (
        '<tempo value="3.5"><loudness value="-20">This chiropractor designed</loudness></tempo> the'
        " device using breakthrough NMES technology."
    ),
    (
        '<tempo value="3.5"><loudness value="-20">Looking for a Strategic'
        " Partner</loudness></tempo> for High-Performance software solutions?"
    ),
    (
        'Depending on your team, <tempo value="3.5"><loudness value="-20">you may also need to'
        " update</loudness></tempo> the Meeting Type."
    ),
]

TEMPO_MAX__LOUDNESS_MAX__CLAUSE = [
    (
        '<tempo value="3.5"><loudness value="17">This trick to get gas for a penny is going to get'
        " banned</loudness></tempo> in Canada."
    ),
    (
        'Let\'s examine <tempo value="3.5"><loudness value="17">what can affect'
        " establishing a positive</loudness></tempo> safety culture."
    ),
    (
        '<tempo value="3.5"><loudness value="17">This chiropractor designed</loudness></tempo> the'
        " device using breakthrough NMES technology."
    ),
    (
        '<tempo value="3.5"><loudness value="17">Looking for a Strategic Partner</loudness></tempo>'
        " for High-Performance software solutions?"
    ),
    (
        'Depending on your team, <tempo value="3.5"><loudness value="17">you may also need to'
        " update</loudness></tempo> the Meeting Type."
    ),
]

LOUDNESS_MIN__TEMPO_MIN__CLAUSE = [
    (
        '<loudness value="-20"><tempo value="0.4">This trick to get gas for a penny is going to get'
        " banned</tempo></loudness> in Canada."
    ),
    (
        'Let\'s examine <loudness value="-20"><tempo value="0.4">what can affect</tempo></loudness>'
        " establishing a positive safety culture."
    ),
    (
        '<loudness value="-20"><tempo value="0.4">This chiropractor designed</tempo></loudness> the'
        " device using breakthrough NMES technology."
    ),
    (
        '<loudness value="-20"><tempo value="0.4">Looking for a Strategic'
        " Partner</tempo></loudness> for High-Performance software solutions?"
    ),
    (
        'Depending on your team, <loudness value="-20"><tempo value="0.4">you may also need to'
        " update</tempo></loudness> the Meeting Type."
    ),
]

LOUDNESS_MIN__TEMPO_MAX__CLAUSE = [
    (
        '<loudness value="-20"><tempo value="3.5">This trick to get gas for a penny is going to get'
        " banned</tempo></loudness> in Canada."
    ),
    (
        'Let\'s examine <loudness value="-20"><tempo value="3.5">what can affect</tempo></loudness>'
        " establishing a positive safety culture."
    ),
    (
        '<loudness value="-20"><tempo value="3.5">This chiropractor designed</tempo></loudness> the'
        " device using breakthrough NMES technology."
    ),
    (
        '<loudness value="-20"><tempo value="3.5">Looking for a Strategic'
        " Partner</tempo></loudness> for High-Performance software solutions?"
    ),
    (
        'Depending on your team, <loudness value="-20"><tempo value="3.5">you may also need to'
        " update</tempo></loudness> the Meeting Type."
    ),
]

LOUDNESS_MAX__TEMPO_MIN__CLAUSE = [
    (
        '<loudness value="17"><tempo value="0.4">This trick to get gas for a penny is going to get'
        " banned</tempo></loudness> in Canada."
    ),
    (
        'Let\'s examine <loudness value="17"><tempo value="0.4">what can affect</tempo></loudness>'
        " establishing a positive safety culture."
    ),
    (
        '<loudness value="17"><tempo value="0.4">This chiropractor designed</tempo></loudness> the'
        " device using breakthrough NMES technology."
    ),
    (
        '<loudness value="17"><tempo value="0.4">Looking for a Strategic Partner</tempo></loudness>'
        " for High-Performance software solutions?"
    ),
    (
        'Depending on your team, <loudness value="17"><tempo value="0.4">you may also need to'
        " update</tempo></loudness> the Meeting Type."
    ),
]

LOUDNESS_MAX__TEMPO_MAX__CLAUSE = [
    (
        '<loudness value="17"><tempo value="3.5">This trick to get gas for a penny is going to get'
        " banned</tempo></loudness> in Canada."
    ),
    (
        'Let\'s examine <loudness value="17"><tempo value="3.5">what can affect</tempo></loudness>'
        " establishing a positive safety culture."
    ),
    (
        '<loudness value="17"><tempo value="3.5">This chiropractor designed</tempo></loudness> the'
        " device using breakthrough NMES technology."
    ),
    (
        '<loudness value="17"><tempo value="3.5">Looking for a Strategic Partner</tempo></loudness>'
        " for High-Performance software solutions?"
    ),
    (
        'Depending on your team, <loudness value="17"><tempo value="3.5">you may also need to'
        " update</tempo></loudness> the Meeting Type."
    ),
]

"""
Tests in DIFFICULT_USER_INITIALISMS,DIFFICULT_USER_QUESTIONS, and DIFFICULT_USER_URLS should be
rendered in v10 as well. These scripts are intended to test the limits of v11 for cases that we
suspect are difficult for v10, and it would be good to have the v10 data to confirm our suspicions.
"""

"""
The following initialisms in context were found by looking at user clips where users had used one
or more workaround methods to input their initialism into WSL Studio. Some of these are presented
with respellings as well as without.
"""
DIFFICULT_USER_INITIALISMS = [
    (
        "Garmin's 7-inch Echomap 73 CV is a dual-beam, dual-frequency chart-plotter that excels at"
        " scanning sonar."
    ),
    (
        "The VS 02 is a bit bulkier, more feature-rich, and more expensive than our previous pick,"
        " but it's easier and less expensive to use, and it also has an automatic shut-off. "
    ),
    (
        "Now, let's discuss LOA Waivers and when it would be appropriate to apply. Anyone who is"
        " not eligible to participate, such as employees out on LOA, should be assigned a waiver."
    ),
    (
        "It provides exceptional adhesion on bare metal and OEM substrates, and it sands easily by"
        " hand or machine."
    ),
    (
        "The next step is to print the necessary sections to PDF so you can include it with the"
        " application packet that will be sent to the client."
    ),
    (
        "A key concept - the LCES system is identified to each firefighter prior to when it must be"
        " used. The nature of wildland fire suppression dictates continuously evaluating and, when"
        " necessary, re-establishing LCES as time and fire growth progress. I want to take a minute"
        " and briefly review each component and its interconnection with the others."
    ),
    "Generate the HTML documentation.",
    "The CMM has completed the inspection.",
    (
        "Next, you can optionally select to Enable WS Trust next to ‘WS Trust Configuration’ if you"
        " have users that log in with Azure AD joined machines, or mail clients that do not support"
        " Modern authentication."
    ),
    (
        "Chatbots: AI-powered chatbots answer frequently asked questions, allowing human"
        " representatives to focus on more complex issues."
    ),
    (
        "Download the My Benefits Work app for"
        " more information. Your Group ID is HPBBAMRG."
    ),
    (
        "Auto GPT is an impressive new feature in Chat GPT that can program AI on its own, making"
        " it both powerful and potentially scary."
    ),
    (
        "Smart LED panels would allow the facility manager to customize, schedule, and remotely"
        " control the building’s lighting with minimal labor and setup when compared to traditional"
        " or standard LED lighting. So, not only could he save energy and reduce costs, but also"
        " improve the overall functionality and convenience of the smart lighting system."
    ),
]

"""
The following questions in context were found by looking at user clips where users demonstrated
frustration in inputting their questions into WSL Studio, such as using multiple question marks
or putting the upward-inflecting final word or phrase in quotation marks.
We are also testing the impact of cues on questions, so some are presented here cued and plain.
Current user feedback: We’re running into a recurring problem with WellSaid - anytime a
script has a question in it, it’s getting flagged as robotic by clients and I have to drop over
to human voice talent. Is there a best practice, or any sort of trick to getting WellSaid voices
to put a little more questioning emphasis at the end of sentences? Right now, it sounds mostly
like a period on the vast majority of avatars when I use default punctuation.
"""
DIFFICULT_USER_QUESTIONS = [
    'Have a topic you’d like us to cover in "Wired In"? Let us know!',
    "Is it similar to configuring the other Contract Types?",
    "Remember our new facility manager?",
    (
        "Hi everyone! I hope the week is going well for you all! Let’s jump into our agenda"
        " starting with an open action. Have the recruiters given Final Approval on the offer"
        " template?"
    ),
    (
        "What did you notice about the shift change at 22:30, when Inspector A filled out his"
        " turnover form and then went home?"
    ),
    (
        'And also she posted a hairstyles picture. "Big Hair? Don\'t care!" With a rolling on the'
        " floor laugh emoji."
    ),
    "Are you ready to learn with CTY?",
    "Perfect! \n\nDo you want to give it a shot for the next Masterdata Contract Type, Jessie?",
    "You mean like getting better drinks at happy hour?",
    (
        "Since the Signatory Roles are a part of the Approval Rule: family, will the Conditions and"
        " Actions be similar to the other Approval Rules?"
    ),
    "Is it fair to say that the root cause of the confusion is in the job-profile titles?",
    "I’m a little busy at the moment. Could we do it sometime later in the week instead?",
    "Will a Business Unit Head be considered as an approver for an agreement at Dynamic Fix?",
    "Danny, is there a way to view the newly created rule on the ICI interface?",
    "Remember our new facility manager?",
    "Is it similar to configuring the other Contract Types?",
    (
        "In what scenario would such restricted access to Attribute Groups be necessary?"
    ),
    (
        "For this, you can configure an Event Rule to add the Secondary Owner to the team, only"
        " when required. You can define multiple conditions when adding a Secondary Owner to the"
        " team, such as an agreement created for a specific vendor, country, specific contract value, "
        " etc. \n\nHas Jeremy mentioned any such specifications?"
    ),
    (
        "I get your point. \n\nOn another note, will it make a difference to the end user whether the attribute "
        " values are retrieved from the choice data type or master data?"
    ),
]
"""
These URLs come directly from user scripts, copied and pasted as found in context. If a URL was
modified (presumably to prompt a better performance from the v10 model), I have also provided a
reconstructed URL in context as a separate test case.
"""
DIFFICULT_USER_URLS = [
    (
        "So, check us out at www.cranetrainingservices.org, or give us a call at 309-231-6146, and"
        " help us unlock your future!"
    ),
    (
        "For the quickest and easiest way to secure your cabin, visit our website at"
        ' www.lifeatseacruises.com and click "Reserve Your Cabin" in the top right corner.'
    ),
    (
        " For immediate assistance or self-help options,"
        " please visit our support page at www.centrios.com/support."
    ),
    "For the latest information go to www.MyTel.com, login, and select MyAccess.",
    (
        " Courtesy the ADA accessibility Guidelines, www.access-board.gov/guides/PDFs/golf.PDF"
        " (see resources below), The US access board is responsible for the development of"
        " minimum-accessibility guidelines for recreation areas, including golf courses."
    ),
    (
        "Savings include 25% off in-house medical services "
        " at participating veterinarians and 25% off purchases from petcareRX.com."
    ),
    (
        "Download the My Benefits Work app, visit www.mybenefitswork.com, or call 800-800-7616 for"
        " more information."
    ),
    "For more information, visit www.BBC.co.uk.",
]

items = locals().items()
V11_TEST_CASES = {
    k: v
    for k, v in items
    if isinstance(v, list) and all(isinstance(t, str) for t in v)
}

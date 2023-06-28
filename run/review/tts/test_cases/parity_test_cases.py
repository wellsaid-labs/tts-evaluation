"""Testing scripts crafted to evaluate TTS Model Parity in performance of question intonation,
initialism pronunciation, and URL pronunciation."""


# Yes/No questions expecting an upward inflection at the end of the sentence
QUESTIONS = [
    "What is the meaning of life, the universe, and everything?",
    "Are birds real?",
    "Would you be a dear and make us a cup of tea?",
    "Must you be so inconsiderate?",
    "Have you considered the implications of faster-than-light travel?",
    "Is it true that the moon is made of cheese?",
    "Does that make sense?",
    "Can I help you?",
    "Are you suggesting that coconuts migrate?",
    "Can I get a backstage pass to the concert?",
    "Given that each of the following bonds is of the same quality and has the same maturity as the others, does one have a longer duration?",
    "Registration of an investment adviser automatically confers registration on?",
    "Church and community leaders in West Virginia, did you know that our state has one of the highest rates of children in foster care per capita in the nation?",
    "Are you daring enough to venture into the unknown and unravel the secrets of our history?",
    "Would your Patients like to receive monthly Kovid-19 at-home test kits at No Cost, delivered straight to their door?",
    "Are you unknowingly damaging your skin, with these common skincare mistakes?",
    "Yes of course, but shall I introduce myself first?",
    "Do you consider yourself an adventurous person?",
    "Have you ever hidden a snack so that nobody else would find it and eat it first?",
    "How do we propose to end the scourge of liens ignorance, you ask?",
]

INITIALISMS = [
    "E3, Electronic Entertainment Expo, is an annual video games conference.",
    "IDK what you mean.",
    "The WHO has declared COVID-19 a global pandemic.",
    "My shady uncle works for the NSA.",
    "NHL allstar Wayne Gretzky played for four teams over the course of his career.",
    "Open up, FBI!",
    "This chart shows the S-P-E software initialization process as it connects to the hardware and transmit is enabled.",
    "There is an MMORPG for almost every major intellectual property.",
    "The 18 BUN dialog box is displayed.",
    "While many Swayj-lock products are classified E.A.R. 99, some Swayj-lock products have ECCNs that require government "
    "licenses, permissions, or other certifications prior to export.",
    "While many Swayj-lock products are classified EAR-99, some Swayj-lock products have ECCNs that require government "
    "licenses, permissions, or other certifications prior to export.",
    "Classification occurs prior to beginning the manufacturing process for ETOP, NPD, and Custom Solutions.",
    "Research tells us that induction process accounts for needs of Aboriginal and CALD young people.",
    "It was a tale of two halves. One half, where Eagles quarterback, Jaylen Hurts, was well on his way to the Super Bowl MVP.",
    "GHG Protocol (or GHGP) was developed in partnership with the World Resources Institute (WRI) "
    "and the World Business Council for Sustainable Development (WBCSD).",
    "Now, this system has generated an A-C-H file, which includes five beneficiaries for which the total amount "
    "of five thousand three hundred ninety-six dollars is being paid.",
    "These sections are located where the smaller IFA's on the narrower left floodplain indicate that flow "
    "in the left floodplain contracts and expands over a shorter distance.",
    "It took six Ph.Ds to design a VCR a five-year-old could use.",
]

URLS = [
    "en.wikipedia.org",
    "linkedin.com",
    "stackoverflow.com",
    "zillow.com",
    "twitch.tv",
    "zoom.us",
    "amazon.de",
    "bbc.co.uk",
    "nih.gov",
    "dailymail.co.uk",
    "imdb.com",
    "nytimes.com",
    "ikea.com",
    "MSDSsource.com",
    "alliancepropertysystems.com",
    "ion2.nokia.com",
]

URLS_HTTPS = [f"https://www.{i}" for i in URLS]

URLS_IN_SENTENCES = [f"Visit our website at {i} for more information." for i in URLS]

URLS_HTTPS_IN_SENTENCES = [f"Visit our website at {i} for more information." for i in URLS_HTTPS]

EMAILS = [
    "tracy.madison@CSWPBC.com",
    "kay_sorensen@insurewithcompass.com",
]

EMAILS_IN_SENTENCES = [f"Please email {i} if you need further support." for i in EMAILS]


V10_PARITY__URLS = URLS_IN_SENTENCES + URLS_HTTPS_IN_SENTENCES + EMAILS_IN_SENTENCES
V10_PARITY__QUESTIONS = QUESTIONS
V10_PARITY__INITIALISMS = INITIALISMS


items = locals().items()
PARITY_TEST_CASES = {
    k: v
    for k, v in items
    if isinstance(v, list) and k.startswith("V10_PARITY") and all(isinstance(t, str) for t in v)
}

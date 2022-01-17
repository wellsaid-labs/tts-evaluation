from functools import partial

from run.data._loader.data_structures import Speaker, make_english_speaker
from run.data._loader.utils import wsl_gcs_dataset_loader

# TODO: Consider not using the actors realnames in the codebase in an effort to protect their
# privacy.

##############
# E-LEARNING #
##############
ADRIENNE_WALKER_HELLER = make_english_speaker(
    label="adrienne_walker_heller", name="Adrienne Walker-Heller"
)
ALICIA_HARRIS = make_english_speaker(label="alicia_harris", name="Alicia Harris")
ALICIA_HARRIS__MANUAL_POST = make_english_speaker(
    label="alicia_harris__manual_post", name="Alicia Harris (Manual Post Processing)"
)
BETH_CAMERON = make_english_speaker(label="beth_cameron", name="Beth Cameron")
BETH_CAMERON__CUSTOM = make_english_speaker(
    label="beth_cameron__custom", name="Beth Cameron (Custom)"
)
ELISE_RANDALL = make_english_speaker(label="elise_randall", name="Elise Randall")
FRANK_BONACQUISTI = make_english_speaker(label="frank_bonacquisti", name="Frank Bonacquisti")
GEORGE_DRAKE_JR = make_english_speaker(label="george_drake_jr", name="George Drake, Jr.")
HANUMAN_WELCH = make_english_speaker(label="hanuman_welch", name="Hanuman Welch")
HEATHER_DOE = make_english_speaker(label="heather_doe", name="Heather Doe")
HILARY_NORIEGA = make_english_speaker(label="hilary_noriega", name="Hilary Noriega")
JACK_RUTKOWSKI = make_english_speaker(label="jack_rutkowski", name="Jack Rutkowski")
JACK_RUTKOWSKI__MANUAL_POST = make_english_speaker(
    label="jack_rutkowski__manual_post", name="Jack Rutkowski (Manual Post Processing)"
)
JOHN_HUNERLACH__NARRATION = make_english_speaker(
    label="john_hunerlach__narration", name="John Hunerlach (Narration)"
)
MARK_ATHERLAY = make_english_speaker(label="mark_atherlay", name="Mark Atherlay")
MEGAN_SINCLAIR = make_english_speaker(label="megan_sinclair", name="Megan Sinclair")
SAM_SCHOLL = make_english_speaker(label="sam_scholl", name="Sam Scholl")
SAM_SCHOLL__MANUAL_POST = make_english_speaker(
    label="sam_scholl__manual_post", name="Sam Scholl (Manual Post Processing)"
)
STEVEN_WAHLBERG = make_english_speaker(label="steven_wahlberg", name="Steven Wahlberg")
SUSAN_MURPHY = make_english_speaker(label="susan_murphy", name="Susan Murphy")

###############
# PROMOTIONAL #
###############
ADRIENNE_WALKER_HELLER__PROMO = make_english_speaker(
    label="adrienne_walker__promo", name="Adrienne Walker-Heller (Promo)"
)
DAMON_PAPADOPOULOS__PROMO = make_english_speaker(
    label="damon_papadopoulos__promo", name="Damon Papadopoulos (Promo)"
)
DANA_HURLEY__PROMO = make_english_speaker(label="dana_hurley__promo", name="Dana Hurley (Promo)")
ED_LACOMB__PROMO = make_english_speaker(label="ed_lacomb__promo", name="Ed LaComb (Promo)")
JOHN_HUNERLACH__RADIO = make_english_speaker(
    label="john_hunerlach__radio", name="John Hunerlach (Radio)"
)
LINSAY_ROUSSEAU__PROMO = make_english_speaker(
    label="linsay_rousseau__promo", name="Linsay Rousseau (Promo)"
)
SAM_SCHOLL__PROMO = make_english_speaker(label="sam_scholl__promo", name="Sam Scholl (Promo)")

#########
# OTHER #
#########
MARI_MONGE__PROMO = make_english_speaker(label="mari_monge__promo", name="Mari Monge (Promo)")
OTIS_JIRY__STORY = make_english_speaker(label="otis_jiry__promo", name="Otis Jiry (Story-Telling)")

#################
# CUSTOM VOICES #
#################
ENERGY_INDUSTRY_ACADEMY__CUSTOM_VOICE = make_english_speaker(
    label="energy_industry_academy__custom_voice", name="Energy Industry Academy (Custom Voice)"
)
THE_EXPLANATION_COMPANY__CUSTOM_VOICE = make_english_speaker(
    label="the_explanation_company__custom_voice", name="The Explanation Company (Custom Voice)"
)

_wsl_speakers = [s for s in locals().values() if isinstance(s, Speaker)]
WSL_DATASETS = {s: partial(wsl_gcs_dataset_loader, speaker=s) for s in _wsl_speakers}

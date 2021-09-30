from functools import partial

from run.data._loader.data_structures import Speaker, make_english_speaker
from run.data._loader.utils import wsl_gcs_dataset_loader

# TODO: Consider not using the actors realnames in the codebase in an effort to protect their
# privacy.

##############
# E-LEARNING #
##############
ADRIENNE_WALKER_HELLER = make_english_speaker("adrienne_walker_heller", "Adrienne Walker-Heller")
ALICIA_HARRIS = make_english_speaker("alicia_harris", "Alicia Harris")
ALICIA_HARRIS__MANUAL_POST = make_english_speaker(
    "alicia_harris__manual_post", "Alicia Harris (Manual Post Processing)"
)
BETH_CAMERON = make_english_speaker("beth_cameron", "Beth Cameron")
BETH_CAMERON__CUSTOM = make_english_speaker("beth_cameron__custom", "Beth Cameron (Custom)")
ELISE_RANDALL = make_english_speaker("elise_randall", "Elise Randall")
FRANK_BONACQUISTI = make_english_speaker("frank_bonacquisti", "Frank Bonacquisti")
GEORGE_DRAKE_JR = make_english_speaker("george_drake_jr", "George Drake, Jr.")
HANUMAN_WELCH = make_english_speaker("hanuman_welch", "Hanuman Welch")
HEATHER_DOE = make_english_speaker("heather_doe", "Heather Doe")
HILARY_NORIEGA = make_english_speaker("hilary_noriega", "Hilary Noriega")
JACK_RUTKOWSKI = make_english_speaker("jack_rutkowski", "Jack Rutkowski")
JACK_RUTKOWSKI__MANUAL_POST = make_english_speaker(
    "jack_rutkowski__manual_post", "Jack Rutkowski (Manual Post Processing)"
)
JOHN_HUNERLACH__NARRATION = make_english_speaker(
    "john_hunerlach__narration", "John Hunerlach (Narration)"
)
MARK_ATHERLAY = make_english_speaker("mark_atherlay", "Mark Atherlay")
MEGAN_SINCLAIR = make_english_speaker("megan_sinclair", "Megan Sinclair")
SAM_SCHOLL = make_english_speaker("sam_scholl", "Sam Scholl")
SAM_SCHOLL__MANUAL_POST = make_english_speaker(
    "sam_scholl__manual_post", "Sam Scholl (Manual Post Processing)"
)
STEVEN_WAHLBERG = make_english_speaker("steven_wahlberg", "Steven Wahlberg")
SUSAN_MURPHY = make_english_speaker("susan_murphy", "Susan Murphy")

###############
# PROMOTIONAL #
###############
ADRIENNE_WALKER_HELLER__PROMO = make_english_speaker(
    "adrienne_walker__promo", "Adrienne Walker-Heller (Promo)"
)
DAMON_PAPADOPOULOS__PROMO = make_english_speaker(
    "damon_papadopoulos__promo", "Damon Papadopoulos (Promo)"
)
DANA_HURLEY__PROMO = make_english_speaker("dana_hurley__promo", "Dana Hurley (Promo)")
ED_LACOMB__PROMO = make_english_speaker("ed_lacomb__promo", "Ed LaComb (Promo)")
JOHN_HUNERLACH__RADIO = make_english_speaker("john_hunerlach__radio", "John Hunerlach (Radio)")
LINSAY_ROUSSEAU__PROMO = make_english_speaker("linsay_rousseau__promo", "Linsay Rousseau (Promo)")
SAM_SCHOLL__PROMO = make_english_speaker("sam_scholl__promo", "Sam Scholl (Promo)")

#########
# OTHER #
#########
MARI_MONGE__PROMO = make_english_speaker("mari_monge__promo", "Mari Monge (Promo)")
OTIS_JIRY__STORY = make_english_speaker("otis_jiry__promo", "Otis Jiry (Story-Telling)")

#################
# CUSTOM VOICES #
#################
ENERGY_INDUSTRY_ACADEMY__CUSTOM_VOICE = make_english_speaker(
    "energy_industry_academy__custom_voice", "Energy Industry Academy (Custom Voice)"
)
THE_EXPLANATION_COMPANY__CUSTOM_VOICE = make_english_speaker(
    "the_explanation_company__custom_voice", "The Explanation Company (Custom Voice)"
)

_wsl_speakers = [s for s in locals().values() if isinstance(s, Speaker)]
WSL_DATASETS = {s: partial(wsl_gcs_dataset_loader, speaker=s) for s in _wsl_speakers}

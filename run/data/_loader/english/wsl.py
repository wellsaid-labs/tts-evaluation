from functools import partial

from run.data._loader.data_structures import Speaker, make_en_speaker
from run.data._loader.utils import wsl_gcs_dataset_loader

# TODO: Consider not using the actors realnames in the codebase in an effort to protect their
# privacy.

##############
# E-LEARNING #
##############
ADRIENNE_WALKER_HELLER = make_en_speaker("adrienne_walker_heller", "Adrienne Walker-Heller")
ALICIA_HARRIS = make_en_speaker("alicia_harris", "Alicia Harris")
ALICIA_HARRIS__MANUAL_POST = make_en_speaker(
    "alicia_harris__manual_post", "Alicia Harris (Manual Post Processing)"
)
BETH_CAMERON = make_en_speaker("beth_cameron", "Beth Cameron")
BETH_CAMERON__CUSTOM = make_en_speaker("beth_cameron__custom", "Beth Cameron (Custom)")
ELISE_RANDALL = make_en_speaker("elise_randall", "Elise Randall")
FRANK_BONACQUISTI = make_en_speaker("frank_bonacquisti", "Frank Bonacquisti")
GEORGE_DRAKE_JR = make_en_speaker("george_drake_jr", "George Drake, Jr.")
HANUMAN_WELCH = make_en_speaker("hanuman_welch", "Hanuman Welch")
HEATHER_DOE = make_en_speaker("heather_doe", "Heather Doe")
HILARY_NORIEGA = make_en_speaker("hilary_noriega", "Hilary Noriega")
JACK_RUTKOWSKI = make_en_speaker("jack_rutkowski", "Jack Rutkowski")
JACK_RUTKOWSKI__MANUAL_POST = make_en_speaker(
    "jack_rutkowski__manual_post", "Jack Rutkowski (Manual Post Processing)"
)
JOHN_HUNERLACH__NARRATION = make_en_speaker(
    "john_hunerlach__narration", "John Hunerlach (Narration)"
)
MARK_ATHERLAY = make_en_speaker("mark_atherlay", "Mark Atherlay")
MEGAN_SINCLAIR = make_en_speaker("megan_sinclair", "Megan Sinclair")
SAM_SCHOLL = make_en_speaker("sam_scholl", "Sam Scholl")
SAM_SCHOLL__MANUAL_POST = make_en_speaker(
    "sam_scholl__manual_post", "Sam Scholl (Manual Post Processing)"
)
STEVEN_WAHLBERG = make_en_speaker("steven_wahlberg", "Steven Wahlberg")
SUSAN_MURPHY = make_en_speaker("susan_murphy", "Susan Murphy")

ALISTAIR_DAVIS__EN_GB = Speaker("alistair_davis__en_gb", "Alistair Davis (en_GB)")

###############
# PROMOTIONAL #
###############
ADRIENNE_WALKER_HELLER__PROMO = make_en_speaker(
    "adrienne_walker__promo", "Adrienne Walker-Heller (Promo)"
)
BRIAN_DIAMOND__EN_IE__PROMO = make_en_speaker(
    "brian_diamond__en_ie__promo", "Brian Diamond (Promo, en_IE)"
)
CHRISTOPHER_DANIELS__PROMO = make_en_speaker(
    "christopher_daniels__promo", "Christopher Daniels (Promo)"
)
DAMON_PAPADOPOULOS__PROMO = make_en_speaker(
    "damon_papadopoulos__promo", "Damon Papadopoulos (Promo)"
)
DAN_FURCA__PROMO = make_en_speaker("dan_furca__promo", "Dan Furca (Promo)")
DANA_HURLEY__PROMO = make_en_speaker("dana_hurley__promo", "Dana Hurley (Promo)")
DARBY_CUPIT__PROMO = make_en_speaker("darby_cupit__promo", "Darby Cupit (Promo)")
ED_LACOMB__PROMO = make_en_speaker("ed_lacomb__promo", "Ed LaComb (Promo)")
IZZY_TUGMAN__PROMO = make_en_speaker("izzy_tugman__promo", "Izzy Tugman (Promo)")
JOHN_HUNERLACH__RADIO = make_en_speaker("john_hunerlach__radio", "John Hunerlach (Radio)")
LINSAY_ROUSSEAU__PROMO = make_en_speaker("linsay_rousseau__promo", "Linsay Rousseau (Promo)")
NAOMI_MERCER_MCKELL__PROMO = make_en_speaker(
    "naomi_mercer_mckell__promo", "Naomi Mercer McKell (Promo)"
)
SAM_SCHOLL__PROMO = make_en_speaker("sam_scholl__promo", "Sam Scholl (Promo)")
SHARON_GAULD_ALEXANDER__PROMO = make_en_speaker(
    "sharon_gauld_alexander__promo", "Sharon Gauld Alexander (Promo)"
)
SHAWN_WILLIAMS__PROMO = make_en_speaker("shawn_williams__promo", "Shawn Williams (Promo)")


#########
# OTHER #
#########
MARI_MONGE__PROMO = make_en_speaker("mari_monge__promo", "Mari Monge (Promo)")
OTIS_JIRY__STORY = make_en_speaker("otis_jiry__promo", "Otis Jiry (Story-Telling)")

#################
# CUSTOM VOICES #
#################
ENERGY_INDUSTRY_ACADEMY__CUSTOM_VOICE = make_en_speaker(
    "energy_industry_academy__custom_voice", "Energy Industry Academy (Custom Voice)"
)
THE_EXPLANATION_COMPANY__CUSTOM_VOICE = make_en_speaker(
    "the_explanation_company__custom_voice", "The Explanation Company (Custom Voice)"
)
HAPPIFY__CUSTOM_VOICE = make_en_speaker("happify__custom_voice", "Happify (Custom Voice)")
SUPER_HI_FI__CUSTOM_VOICE = make_en_speaker(
    "super_hi_fi__custom_voice", "Super HiFi (Custom Voice)"
)
US_PHARMACOPEIA__CUSTOM_VOICE = make_en_speaker(
    "us_pharmacopeia__custom_voice", "US Pharmacopeia (Custom Voice)"
)
VERITONE__CUSTOM_VOICE = make_en_speaker("veritone__custom_voice", "Veritone (Custom Voice)")
VIACOM__CUSTOM_VOICE = make_en_speaker("viacom__custom_voice", "Viacom (Custom Voice)")
HOUR_ONE_NBC__BB_CUSTOM_VOICE = make_en_speaker(
    "hour_one_nbc__bb_custom_voice", "HourOne X NBC (BB Custom Voice)"
)

_wsl_speakers = [s for s in locals().values() if isinstance(s, Speaker)]
WSL_DATASETS = {s: partial(wsl_gcs_dataset_loader, speaker=s) for s in _wsl_speakers}

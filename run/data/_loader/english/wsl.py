from functools import partial

from run.data._loader import structures as struc
from run.data._loader.utils import wsl_gcs_dataset_loader

# TODO: Consider not using the actors realnames in the codebase in an effort to protect their
# privacy.

Dia = struc.Dialect


def make(
    label: str,
    name: str,
    gcs_dir: str,
    dialect: struc.Dialect = Dia.EN_US,
    style: struc.Style = struc.Style.NARR,
    post: bool = False,
) -> struc.Speaker:
    return struc.Speaker(label, style, dialect, name, gcs_dir, post)


##############
# E-LEARNING #
##############
narr = partial(make, style=struc.Style.NARR)
SOFIA_H = narr("sofia_h", "Sofia H-Heller", "adrienne_walker_heller")
AVA_M = narr("ava_m", "Ava M", "alicia_harris")
AVA_M__MANUAL_POST = narr(
    AVA_M.label,
    "Ava M (Manual Post Processing)",
    "alicia_harris__manual_post",
    post=True,
)
RAMONA_J = narr("ramona_j", "Ramona J", "beth_cameron")
PAIGE_L = narr("paige_l", "Paige L", "elise_randall")
DAVID_D = narr("david_d", "David D", "frank_bonacquisti")
JEREMY_G = narr("jeremy_g", "Jeremy G", "george_drake_jr")
TOBIN_A = narr("tobin_a", "Tobin A", "hanuman_welch")
ISABEL_V = narr("isabel_v", "Heather Doe", "heather_doe")
ALANA_B = narr("alana_b", "Alana B", "hilary_noriega")
KAI_M = narr("kai_m", "Kai M", "jack_rutkowski")
KAI_M__MANUAL_POST = narr(
    KAI_M.label,
    "Kai M (Manual Post Processing)",
    "jack_rutkowski__manual_post",
    post=True,
)
JOE_F__NARRATION = narr("joe_f", "Joe F (Narration)", "john_hunerlach__narration")
TRISTAN_F = narr("tristan_f", "Tristan F", "mark_atherlay")
NICOLE_L = narr("nicole_l", "Nicole L", "megan_sinclair")
WADE_C = narr("wade_c", "Wade C", "sam_scholl")
WADE_C__MANUAL_POST = narr(
    WADE_C.label,
    "Wade C (Manual Post Processing)",
    "sam_scholl__manual_post",
    post=True,
)
PATRICK_K = narr("patrick_k", "Patrick K", "steven_wahlberg")
VANESSA_N = narr("vanessa_n", "Vanessa N", "susan_murphy")
JUDE_D__EN_GB = narr(
    "jude_d__en_gb",
    "Jude D (British English)",
    "alistair_davis__en_gb",
    Dia.EN_UK,
)

# 2021 Q1 NARRATION
GIA_V = narr("gia_v", "Gia V", "alessandra_ruiz")
ANTONY_A = narr("antony_a", "Antony A", "alex_marrero")
JARVIS_H = narr("jarvis_h", "Alexander Hill Knight (en_GB)", "alexander_hill_knight", Dia.EN_UK)
JODI_P = narr("jodi_p", "Jodi P", "dana_hurley")
RAINE_B = narr("raine_b", "Raine B", "diontae_black")
OWEN_C = narr("owen_c", "Owen C", "marc_white")
THEO_K = narr("theo_k", "Theo K (en_AU)", "piotr_kohnke", Dia.EN_AU)
ZACH_E = narr("zach_e", "Zach E", "seth_jones")
GENEVIEVE_M = narr("genevieve_m", "Genevieve M", "sophie_reppert")
JAMES_B = narr("james_b", "James B (en_GB)", "steve_newman", Dia.EN_UK)

###############
# PROMOTIONAL #
###############
promo = partial(make, style=struc.Style.PROMO)
SOFIA_H__PROMO = promo(SOFIA_H.label, "Sofia H-Heller (Promo)", "adrienne_walker__promo")
ERIC_S__EN_IE__PROMO = promo(
    "eric_s__en_ie",
    "Eric S (Promo, Ireland)",
    "brian_diamond__en_ie__promo",
    Dia.EN_IE,
)
CHASE_J__PROMO = promo("chase_j", "Chase J (Promo)", "christopher_daniels__promo")
DAMIAN_P__PROMO = promo("damian_p", "Damian P (Promo)", "damon_papadopoulos__promo", Dia.EN_CA)
JODI_P__PROMO = promo(JODI_P.label, "Jodi P (Promo)", "dana_hurley__promo")
STEVE_B__PROMO = promo("steve_b", "Steve B (Promo)", "darby_cupit__promo")
LEE_M__PROMO = promo("lee_m", "Lee M (Promo)", "ed_lacomb__promo")
BELLA_B__PROMO = promo("bella_b", "Bella B (Promo)", "izzy_tugman__promo")
SELENE_R__PROMO = promo("selene_r", "Selene R (Promo)", "linsay_rousseau__promo")
TILDA_C__PROMO = promo("tilda_c", "Tilda C (Promo)", "naomi_mercer_mckell__promo")
WADE_C__PROMO = promo(WADE_C.label, "Wade C (Promo)", "sam_scholl__promo")
PAUL_B__PROMO = promo("paul_b", "Paul B (Promo)", "shawn_williams__promo")
SHARON_GAULD_ALEXANDER__PROMO = promo(
    "sharon_gauld_alexander",
    "Sharon Gauld Alexander (Promo)",
    "sharon_gauld_alexander__promo",
    Dia.EN_CA,
)

##################
# CONVERSATIONAL #
##################
convo = partial(make, style=struc.Style.CONVO)
SOFIA_H__CONVO = convo(SOFIA_H.label, "Sofia H (Convo)", "adrienne_walker__convo")
AVA_M__CONVO = convo(AVA_M.label, "Ava M (Convo)", "alicia_harris__convo")
KAI_M__CONVO = convo(KAI_M.label, "Kai M (Convo)", "jack_rutkowski__convo")
NICOLE_L__CONVO = convo(NICOLE_L.label, "Nicole L (Convo)", "megan_sinclair__convo")
WADE_C__CONVO = convo(WADE_C.label, "Wade C (Convo)", "sam_scholl__convo")
PATRICK_K__CONVO = convo(PATRICK_K.label, "Patrick K (Convo)", "steven_wahlberg__convo")
VANESSA_N__CONVO = convo(VANESSA_N.label, "Vanessa N (Convo)", "susan_murphy__convo")

#########
# OTHER #
#########
other = partial(make, style=struc.Style.OTHER)
MARI_MONGE__PROMO = other("mari_monge", "Mari Monge (Promo)", "mari_monge__promo")
DAN_FURCA__PROMO = other("dan_furca", "Dan Furca (Promo)", "dan_furca__promo")
GARRY_J__STORY = other("garry_j", "Garry J (Story-Telling)", "otis_jiry__promo")
RAMONA_J__CUSTOM = other(RAMONA_J.label, "Ramona J (Custom)", "beth_cameron__custom")
JOE_F__RADIO = promo(JOE_F__NARRATION.label, "Joe F (Radio)", "john_hunerlach__radio")


#################
# CUSTOM VOICES #
#################
custom = lambda gcs_dir, name, dia=Dia.EN_US: make(
    gcs_dir, f"{name}", gcs_dir, dia, style=struc.Style.OTHER
)
UNEEQ__ASB_CUSTOM_VOICE = custom(
    "uneeq__asb_custom_voice",
    "UneeQ - ASB (Custom Voice V2)",
    dia=Dia.EN_NZ,
)
# UneeQ Custom Voice, built via combining V1 and V2 datasets, normalized to 21 LUFS
UNEEQ__ASB_CUSTOM_VOICE_COMBINED = custom(
    "uneeq__asb__combined_data_21_lufs",
    "UneeQ - ASB (Custom Voice V3)",
    dia=Dia.EN_NZ,
)
ENERGY_INDUSTRY_ACADEMY__CUSTOM_VOICE = custom(
    "energy_industry_academy__custom_voice",
    "Energy Industry Academy (Custom Voice)",
)
THE_EXPLANATION_COMPANY__CUSTOM_VOICE = custom(
    "the_explanation_company__custom_voice",
    "The Explanation Company (Custom Voice)",
)
HAPPIFY__CUSTOM_VOICE = custom("happify__custom_voice", "Happify (Custom Voice)")
SUPER_HI_FI__CUSTOM_VOICE = custom("super_hi_fi__custom_voice", "Super HiFi (Custom Voice)")
US_PHARMACOPEIA__CUSTOM_VOICE = custom(
    "us_pharmacopeia__custom_voice",
    "US Pharmacopeia (Custom Voice)",
)
VERITONE__CUSTOM_VOICE = custom("veritone__custom_voice", "Veritone (Custom Voice)")
VIACOM__CUSTOM_VOICE = custom("viacom__custom_voice", "Viacom (Custom Voice)")
HOUR_ONE_NBC__BB_CUSTOM_VOICE = custom(
    "hour_one_nbc__bb_custom_voice",
    "HourOne X NBC (BB Custom Voice)",
)
STUDY_SYNC__CUSTOM_VOICE = custom("study_sync__custom_voice", "StudySync (Custom Voice)")
FIVE_NINE__CUSTOM_VOICE = custom("five_nine__custom_voice", "Five9 (Custom Voice)")

_wsl_speakers = [s for s in locals().values() if isinstance(s, struc.Speaker)]
WSL_DATASETS = {s: partial(wsl_gcs_dataset_loader, speaker=s) for s in _wsl_speakers}

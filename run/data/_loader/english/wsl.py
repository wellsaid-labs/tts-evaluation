from functools import partial

from run.data._loader import structures as struc
from run.data._loader.utils import wsl_gcs_dataset_loader

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
narr = partial(make, style=struc.Style.OG_NARR)
# Cleaned 04/2023:
SOFIA_H = narr("sofia_h", "Sofia H (Narration V2)", "sofia_h__narration__v2")
AVA_M = narr("ava_m", "Ava M", "alicia_harris")
# Cleaned 04/2023:
AVA_M__NARRATION = narr(AVA_M.label, "Ava M (Narration V3)", "ava_m__narration__v3")
RAMONA_J = narr("ramona_j", "Ramona J", "beth_cameron")
PAIGE_L = narr("paige_l", "Paige L", "elise_randall")
DAVID_D = narr("david_d", "David D", "frank_bonacquisti")
JEREMY_G = narr("jeremy_g", "Jeremy G", "george_drake_jr")
TOBIN_A = narr("tobin_a", "Tobin A", "hanuman_welch")
# Cleaned 04/2023:
ISABEL_V = narr("isabel_v", "Isabel V (Narration V2)", "isabel_v__narration__v2")
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
# Cleaned 04/2023:
WADE_C__NARRATION = narr(WADE_C.label, "Wade C (Narration V3)", "wade_c__narration__v3")
PATRICK_K = narr("patrick_k", "Patrick K", "steven_wahlberg")
VANESSA_N = narr("vanessa_n", "Vanessa N", "susan_murphy")

# 2021 Q1 NARRATION
narr = partial(make, style=struc.Style.NARR)
JUDE_D__EN_GB = narr(
    "jude_d__en_gb",
    "Jude D (British English)",
    "alistair_davis__en_gb",
    Dia.EN_ZA,
)
GIA_V = narr("gia_v", "Gia V", "alessandra_ruiz")
ANTONY_A = narr("antony_a", "Antony A", "alex_marrero")
# TODO: Update `en_GB` to `en_UK` for consistency.
JARVIS_H = narr("jarvis_h", "Alexander Hill Knight (en_GB)", "alexander_hill_knight", Dia.EN_UK)
JODI_P = narr("jodi_p", "Jodi P", "dana_hurley")
OWEN_C = narr("owen_c", "Owen C", "marc_white")
THEO_K = narr("theo_k", "Theo K (en_AU)", "piotr_kohnke", Dia.EN_AU)
ZACH_E = narr("zach_e", "Zach E", "seth_jones")
GENEVIEVE_M = narr("genevieve_m", "Genevieve M", "sophie_reppert")
# TODO: Update `en_GB` to `en_UK` for consistency.
# Cleaned 04/2023:
JAMES_B = narr("james_b", "James B (Narration V2, UK)", "james_b__narration__v2", Dia.EN_UK)

# 2022 Q2 NARRATION
# Cleaned 04/2023:
TERRA_G = narr("terra_g", "Terra G (Narration V2)", "terra_g__narration__v2")
PHILIP_J = narr("philip_j", "Chris Anderson", "chris_anderson")
MARCUS_G = narr("marcus_g", "Dan Furca", "dan_furca")
JORDAN_T = narr("jordan_t", "Danielle Whiteside", "danielle_whiteside")
DONNA_W = narr("donna_w", "Erica Brookhyser (en_US, Southern)", "erica_brookhyser")
# Cleaned 04/2023:
GREG_G = narr("greg_g", "Greg G (Narration V2, AU))", "greg_g__narration__en_au__v2", Dia.EN_AU)
ZOEY_O = narr("zoey_o", "Helen Marion Rowe (en_AU)", "helen_marion_rowe", Dia.EN_AU)
# Cleaned 04/2023:
FIONA_H = narr("fiona_h__en_uk", "Donnla Hughes (en_UK)", "donnla_hughes", Dia.EN_UK)
ROXY_T = narr("roxy_t", "Emma Topping (en_UK)", "emma_topping", Dia.EN_UK)
KARI_N = narr("kari_n", "Kara Noble (en_UK)", "kara_noble", Dia.EN_UK)
# Cleaned 04/2023:
DIARMID_C = narr("diarmid_c", "Diarmid C (en_UK)", "diarmid_c__narration__en_uk__v2", Dia.EN_UK)
ELIZABETH_U = narr("elizabeth_u", "Suzi Stringer (en_UK)", "suzi_stringer")
ALAN_T = narr("alan_t", "Tomas Frazer (en_UK)", "tomas_frazer", Dia.EN_UK)

# 2022 Q3 NARRATION
GRAY_L = narr("gray_l", "Gray L", "tawny_platis")
MICHAEL_V = narr("michael_v", "Michael V", "keith_forsgren")
# Cleaned 04/2023:
PAULA_R = narr("paula_r", "Paula R (en_MX)", "paula_r", Dia.EN_MX)
BEN_D = narr("ben_d", "Ben D (en_ZA)", "daniel_barnett", Dia.EN_ZA)
BELLA_B = narr("bella_b", "Bella B", "izzy_tugman")

# 2022 Q4 NARRATION
# Cleaned 04/2023:
ABBI_D = narr("abbi_d", "Abbi D (Narration V2)", "abbi_d__narration__v2")
LULU_G = narr("lulu_g", "Lulu G", "lulu_g")
FIONA_H_IE = narr("fiona_h__en_ie", "Fiona H (en_IE)", "fiona_h", Dia.EN_IE)
LORENZO_D = narr("lorenzo_d", "Lorenzo D (en_MX)", "lorenzo_d", Dia.EN_MX)
HANNAH_A = narr("hannah_a", "Hannah A (en_MX)", "hannah_a", Dia.EN_MX)
OLIVER_S = narr("oliver_s", "Oliver S (en_UK)", "oliver_s", Dia.EN_UK)
JACK_C = narr("jack_c", "Jack C", "jack_c")
JENSEN_X = narr("jensen_x", "Jensen X", "jensen_x")
ERIC_S = narr("eric_s", "Eric S (en_IE)", "eric_s", Dia.EN_IE)

# 2023 Q1 NARRATION
SHELBY_D = narr("shelby_d", "Shelby D", "shelby_d")
SE_VON_M = narr("se_von_m", "Se'von M", "se_von_m")
JIMMY_J = narr("jimmy_j", "Jimmy J", "jimmy_j")
JAY_S = narr("jay_s", "Jay S", "jay_s")
# Cleaned 04/2023:
RAINE_B = narr("raine_b", "Raine B (Narration V3)", "raine_b__narration__v3")
SELENE_R = narr("selene_r", "Selene R", "selene_r")
ISSA_B = narr("issa_b", "Issa B (en_ZA)", "issa_b", Dia.EN_ZA)
LYRIC_K = narr("lyric_k", "Lyric K", "lyric_k")
ALI_P = narr("ali_p", "Ali P (en_AU)", "ali_p", Dia.EN_AU)
AARON_G = narr("aaron_g", "Aaron G (en_AU)", "aaron_g", Dia.EN_AU)

###############
# PROMOTIONAL #
###############
promo = partial(make, style=struc.Style.PROMO)
# Cleaned 04/2023:
SOFIA_H__PROMO = promo(SOFIA_H.label, "Sofia H (Promo V2)", "sofia_h__promo__v2")
# Cleaned 04/2023:
AVA_M__PROMO = promo(AVA_M.label, "Ava M (Promo, V2)", "ava_m__promo__v2")
ERIC_S__EN_IE__PROMO = promo(
    "eric_s__en_ie",
    "Eric S (Promo, Ireland)",
    "brian_diamond__en_ie__promo",
    Dia.EN_IE,
)
CHASE_J__PROMO = promo("chase_j", "Chase J (Promo)", "christopher_daniels__promo")
# Cleaned 04/2023:
DAMIAN_P__PROMO = promo("damian_p", "Damian P (Promo, V2)", "damian_p__promo__en_ca__v2", Dia.EN_CA)
JODI_P__PROMO = promo(JODI_P.label, "Jodi P (Promo)", "dana_hurley__promo")
STEVE_B__PROMO = promo("steve_b", "Steve B (Promo)", "darby_cupit__promo")
# Cleaned 04/2023:
LEE_M__PROMO = promo("lee_m", "Lee M (Promo V2)", "lee_m__promo__v2")
BELLA_B__PROMO = promo("bella_b", "Bella B (Promo)", "izzy_tugman__promo")
SELENE_R__PROMO = promo("selene_r", "Selene R (Promo)", "linsay_rousseau__promo")
TILDA_C__PROMO = promo("tilda_c", "Tilda C (Promo)", "naomi_mercer_mckell__promo")
# Cleaned 04/2023:
WADE_C__PROMO = promo(WADE_C.label, "Wade C (Promo V2)", "wade_c__promo__v2")
PAUL_B__PROMO = promo("paul_b", "Paul B (Promo)", "shawn_williams__promo")
CHARLIE_Z__PROMO = promo(
    "charlie_z", "Sharon Gauld Alexander (Promo)", "sharon_gauld_alexander__promo", Dia.EN_CA
)

# 2022 Q4 Promotional
# Cleaned 04/2023:
TOBIN_A__PROMO = promo(TOBIN_A.label, "Tobin A (Promo V3)", "tobin_a__promo__v3")
# Cleaned 04/2023:
DIARMID_C__PROMO = promo(
    DIARMID_C.label, "Diarmid C (Promo V2, UK)", "diarmid_c__promo__en_uk__v2", Dia.EN_UK
)
# Cleaned 04/2023:
JARVIS_H__PROMO = promo(
    JARVIS_H.label, "Jarvis H (Promo V2, UK)", "jarvis_h__promo__en_uk__v2", Dia.EN_UK
)
GIA_V__PROMO = promo(GIA_V.label, "Gia V (Promo)", "gia_v__promo")
OWEN_C__PROMO = promo(OWEN_C.label, "Owen C (Promo)", "owen_c__promo")

# 2023 Q1 Promotional
JORDAN_T__PROMO = promo(JORDAN_T.label, "Jordan T (Promo)", "jordan_t__promo")
# Cleaned 04/2023:
GREG_G__PROMO = promo(GREG_G.label, "Greg G (Promo V2, AU)", "greg_g__promo__en_au__v2", Dia.EN_AU)
GENEVIEVE_M__PROMO = promo(GENEVIEVE_M.label, "Genevieve M (Promo)", "genevieve_m__promo")
MARCUS_G_PROMO = promo(MARCUS_G.label, "Marcus G (Promo)", "marcus_g__promo__v2")
NICOLE_L__PROMO = promo(NICOLE_L.label, "Nicole L (Promo)", "nicole_l__promo")
PHILIP_J__PROMO = promo(PHILIP_J.label, "Philip J (Promo)", "philip_j__promo")
VANESSA_N__PROMO = promo(VANESSA_N.label, "Vanessa N (Promo)", "vanessa_n__promo")

##################
# CONVERSATIONAL #
##################
convo = partial(make, style=struc.Style.CONVO)
# Cleaned 04/2023:
SOFIA_H__CONVO = convo(SOFIA_H.label, "Sofia H (Convo V2)", "sofia_h__convo__v2")
# Cleaned 04/2023:
AVA_M__CONVO = convo(AVA_M.label, "Ava M (Convo, V2)", "ava_m__convo__v2")
TOBIN_A__CONVO = convo(TOBIN_A.label, "Hanuman Welch (Convo)", "hanuman_welch__convo")
KAI_M__CONVO = convo(KAI_M.label, "Kai M (Convo)", "jack_rutkowski__convo")
NICOLE_L__CONVO = convo(NICOLE_L.label, "Nicole L (Convo)", "megan_sinclair__convo")
# Cleaned 04/2023:
WADE_C__CONVO = convo(WADE_C.label, "Wade C (Convo V2)", "wade_c__convo__v2")
PATRICK_K__CONVO = convo(PATRICK_K.label, "Patrick K (Convo)", "steven_wahlberg__convo")
VANESSA_N__CONVO = convo(VANESSA_N.label, "Vanessa N (Convo)", "susan_murphy__convo")
MARCUS_G__CONVO = convo(MARCUS_G.label, "Marcus G (Convo)", "marcus_g__convo")

# 2022 Q4 Conversational
JORDAN_T__CONVO = convo(JORDAN_T.label, "Jordan T (Convo)", "danielle_whiteside__convo")
JODI_P__CONVO = convo(JODI_P.label, "Jodi P (Convo)", "jodi_p__convo")
# Cleaned 04/2023:
JARVIS_H__CONVO = convo(
    JARVIS_H.label, "Jarvis H (Convo V2, UK)", "jarvis_h__convo__en_uk__v2", Dia.EN_UK
)
GIA_V__CONVO = convo(GIA_V.label, "Gia V (Convo)", "gia_v__convo")
OWEN_C__CONVO = convo(OWEN_C.label, "Owen C (Convo)", "owen_c__convo")
PHILIP_J__CONVO = convo(PHILIP_J.label, "Philip J (Convo)", "philip_j__convo")
ANTONY_A__CONVO = convo(ANTONY_A.label, "Antony A (Convo)", "antony_a__convo")
BELLA_B__CONVO = convo(BELLA_B.label, "Bella B (Convo)", "bella_b__convo")
# Cleaned 04/2023:
GREG_G__CONVO = convo(GREG_G.label, "Greg G (Convo V2, AU)", "greg_g__convo__en_au__v2", Dia.EN_AU)
GENEVIEVE_M__CONVO = convo(GENEVIEVE_M.label, "Genevieve M (Convo)", "genevieve_m__convo")

# 2023 Q1 Conversational
DIARMID_C__CONVO = convo(
    DIARMID_C.label, "Diarmid C (Convo V2, UK)", "diarmid_c__convo__en_uk__v2", Dia.EN_UK
)
JAMES_B__CONVO = convo(JAMES_B.label, "James B (Convo V2, UK)", "james_b__convo__v2", Dia.EN_UK)
LEE_M__CONVO = convo(LEE_M__PROMO.label, "Lee M (Convo V2)", "lee_m__convo__v2")
TILDA_C__CONVO = convo(TILDA_C__PROMO.label, "Tilda C (Convo)", "tilda_c__convo")

#########
# OTHER #
#########
other = partial(make, style=struc.Style.OTHER)
MARI_MONGE__PROMO = other("mari_monge", "Mari Monge (Promo)", "mari_monge__promo")
DAN_FURCA__PROMO = other("dan_furca", "Dan Furca (Promo)", "dan_furca__promo")
GARRY_J__STORY = other("garry_j", "Garry J (Story-Telling)", "otis_jiry__promo")
RAMONA_J__CUSTOM = other(RAMONA_J.label, "Ramona J (Custom)", "beth_cameron__custom")
JOE_F__RADIO = promo(JOE_F__NARRATION.label, "Joe F (Radio)", "john_hunerlach__radio")

# 2023 Q1 Character
KARI_N__CHARACTER = other("kari_n", "Kari N (Character, UK)", "kari_n__character", Dia.EN_UK)

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
    "energy_industry_academy__tailored_script",
    "Energy Industry Academy (Custom Voice, Tailored Script)",
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
STUDY_SYNC__CUSTOM_VOICE = custom("studysync__custom_voice", "StudySync (Custom Voice)")
FIVE_NINE__CUSTOM_VOICE = custom("fivenine__custom_voice", "Five9 (Custom Voice)")
SELECTQUOTE__CUSTOM_VOICE = custom("select_quote__custom_voice", "SelectQuote (Custom Voice)")

_wsl_speakers = [s for s in locals().values() if isinstance(s, struc.Speaker)]
WSL_DATASETS = {s: partial(wsl_gcs_dataset_loader, speaker=s) for s in _wsl_speakers}

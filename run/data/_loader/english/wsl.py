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
# Previously miscategorized as promo
CHASE_J = narr("chase_j", "Chase J (Narration)", "christopher_daniels__promo")
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
RAINE_B = narr("raine_b", "Raine B", "diontae_black")
OWEN_C = narr("owen_c", "Owen C", "marc_white")
THEO_K = narr("theo_k", "Theo K (en_AU)", "piotr_kohnke", Dia.EN_AU)
ZACH_E = narr("zach_e", "Zach E", "seth_jones")
GENEVIEVE_M = narr("genevieve_m", "Genevieve M", "sophie_reppert")
# TODO: Update `en_GB` to `en_UK` for consistency.
JAMES_B = narr("james_b", "James B (en_GB)", "steve_newman", Dia.EN_UK)

# 2022 Q2 NARRATION
TERRA_G = narr("terra_g", "Celeste Parrish", "celeste_parrish")
PHILIP_J = narr("philip_j", "Chris Anderson", "chris_anderson")
MARCUS_G = narr("marcus_g", "Dan Furca", "dan_furca")
JORDAN_T = narr("jordan_t", "Danielle Whiteside", "danielle_whiteside")
DONNA_W = narr("donna_w", "Erica Brookhyser (en_US, Southern)", "erica_brookhyser")
GREG_G = narr("greg_g", "Glen Lloyd (en_AU)", "glen_lloyd", Dia.EN_AU)
ZOEY_O = narr("zoey_o", "Helen Marion Rowe (en_AU)", "helen_marion_rowe", Dia.EN_AU)
FIONA_H = narr("fiona_h", "Donnla Hughes (en_UK)", "donnla_hughes", Dia.EN_UK)
ROXY_T = narr("roxy_t", "Emma Topping (en_UK)", "emma_topping", Dia.EN_UK)
KARI_N = narr("kari_n", "Kara Noble (en_UK)", "kara_noble", Dia.EN_UK)
DIARMID_C = narr("diarmid_c", "Kevin Cherry (en_UK, Scottish)", "kevin_cherry", Dia.EN_UK)
ELIZABETH_U = narr("elizabeth_u", "Suzi Stringer (en_UK)", "suzi_stringer")
ALAN_T = narr("alan_t", "Tomas Frazer (en_UK)", "tomas_frazer", Dia.EN_UK)

# 2022 Q3 NARRATION
GRAY_L = narr("gray_l", "Gray L", "tawny_platis")
MICHAEL_V = narr("michael_v", "Michael V", "keith_forsgren")
PAULA_R = narr("paula_r", "Paula R", "paula_r")
BEN_D = narr("ben_d", "Ben D (en_ZA)", "daniel_barnett", Dia.EN_ZA)
BELLA_B = narr("bella_b", "Bella B", "izzy_tugman")

# 2022 Q4 NARRATION
ABBI_D = narr("abbi_d", "Abbi D", "abbi_d")
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
SELENE_R = narr("selene_r", "Selene R", "selene_r")
ISSA_B = narr("issa_b", "Issa B (en_ZA)", "issa_b", Dia.EN_ZA)
LYRIC_K = narr("lyric_k", "Lyric K", "lyric_k")
ALI_P = narr("ali_p", "Ali P (en_AU)", "ali_p", Dia.EN_AU)
AARON_G = narr("aaron_g", "Aaron G (en_AU)", "aaron_g", Dia.EN_AU)

###############
# PROMOTIONAL #
###############
promo = partial(make, style=struc.Style.PROMO)
SOFIA_H__PROMO = promo(SOFIA_H.label, "Sofia H-Heller (Promo)", "adrienne_walker__promo")
AVA_M__PROMO = promo(AVA_M.label, "Alicia Harris (Promo)", "alicia_harris__promo")
# Previously miscategorized as narration
ZACH_E__PROMO = promo("zach_e", "Zach E", "seth_jones")
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
TOBIN_A__PROMO = promo(TOBIN_A.label, "Hanuman Welch (Promo)", "hanuman_welch__promo")
BELLA_B__PROMO = promo("bella_b", "Bella B (Promo)", "izzy_tugman__promo")
SELENE_R__PROMO = promo("selene_r", "Selene R (Promo)", "linsay_rousseau__promo")
TILDA_C__PROMO = promo("tilda_c", "Tilda C (Promo)", "naomi_mercer_mckell__promo")
WADE_C__PROMO = promo(WADE_C.label, "Wade C (Promo)", "sam_scholl__promo")
PAUL_B__PROMO = promo("paul_b", "Paul B (Promo)", "shawn_williams__promo")
CHARLIE_Z__PROMO = promo(
    "charlie_z", "Sharon Gauld Alexander (Promo)", "sharon_gauld_alexander__promo", Dia.EN_CA
)

# 2022 Q4 Promotional
DIARMID_C__PROMO = promo(DIARMID_C.label, "Diarmid C (Promo, UK)", "diarmid_c__promo", Dia.EN_UK)
JARVIS_H__PROMO = promo(JARVIS_H.label, "Jarvis H (Promo, UK)", "jarvis_h__promo", Dia.EN_UK)
GIA_V__PROMO = promo(GIA_V.label, "Gia V (Promo)", "gia_v__promo")
OWEN_C__PROMO = promo(OWEN_C.label, "Owen C (Promo)", "owen_c__promo")

# 2023 Q1 Promotional
JORDAN_T__PROMO = promo(JORDAN_T.label, "Jordan T (Promo)", "jordan_t__promo")
GENEVIEVE_M__PROMO = promo(GENEVIEVE_M.label, "Genevieve M (Promo)", "genevieve_m__promo")
NICOLE_L__PROMO = promo(NICOLE_L.label, "Nicole L (Promo)", "nicole_l__promo")
PHILIP_J__PROMO = promo(PHILIP_J.label, "Philip J (Promo)", "philip_j__promo")
VANESSA_N__PROMO = promo(VANESSA_N.label, "Vanessa N (Promo)", "vanessa_n__promo")

##################
# CONVERSATIONAL #
##################
convo = partial(make, style=struc.Style.CONVO)
SOFIA_H__CONVO = convo(SOFIA_H.label, "Sofia H (Convo)", "adrienne_walker__convo")
AVA_M__CONVO = convo(AVA_M.label, "Ava M (Convo)", "alicia_harris__convo")
TOBIN_A__CONVO = convo(TOBIN_A.label, "Hanuman Welch (Convo)", "hanuman_welch__convo")
KAI_M__CONVO = convo(KAI_M.label, "Kai M (Convo)", "jack_rutkowski__convo")
NICOLE_L__CONVO = convo(NICOLE_L.label, "Nicole L (Convo)", "megan_sinclair__convo")
WADE_C__CONVO = convo(WADE_C.label, "Wade C (Convo)", "sam_scholl__convo")
PATRICK_K__CONVO = convo(PATRICK_K.label, "Patrick K (Convo)", "steven_wahlberg__convo")
VANESSA_N__CONVO = convo(VANESSA_N.label, "Vanessa N (Convo)", "susan_murphy__convo")
MARCUS_G__CONVO = convo(MARCUS_G.label, "Marcus G (Convo)", "marcus_g__convo")

# 2022 Q4 Conversational
JORDAN_T__CONVO = convo(JORDAN_T.label, "Jordan T (Convo)", "danielle_whiteside__convo")
JODI_P__CONVO = convo(JODI_P.label, "Jodi P (Convo)", "jodi_p__convo")
JARVIS_H__CONVO = convo(JARVIS_H.label, "Jarvis H (Convo, UK)", "jarvis_h__convo", Dia.EN_UK)
GIA_V__CONVO = convo(GIA_V.label, "Gia V (Convo)", "gia_v__convo")
OWEN_C__CONVO = convo(OWEN_C.label, "Owen C (Convo)", "owen_c__convo")
PHILIP_J__CONVO = convo(PHILIP_J.label, "Philip J (Convo)", "philip_j__convo")
ANTONY_A__CONVO = convo(ANTONY_A.label, "Antony A (Convo)", "antony_a__convo")
BELLA_B__CONVO = convo(BELLA_B.label, "Bella B (Convo)", "bella_b__convo")
GREG_G__CONVO = convo(GREG_G.label, "Greg G (Convo, AU)", "greg_g__convo", Dia.EN_AU)
GENEVIEVE_M__CONVO = convo(GENEVIEVE_M.label, "Genevieve M (Convo)", "genevieve_m__convo")

# 2023 Q1 Conversational
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

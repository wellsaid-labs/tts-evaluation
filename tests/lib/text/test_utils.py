import re
import string
import typing
from typing import get_args
from unittest import mock

import pytest

import lib
from lib.text.utils import (
    RESPELLING_ALPHABET,
    RESPELLINGS,
    _remove_arpabet_markings,
    grapheme_to_phoneme,
    is_normalized_vo_script,
    normalize_vo_script,
)


def test_grapheme_to_phoneme():
    """Test `grapheme_to_phoneme` can handle some basic cases."""
    in_ = [
        "  Hello World  ",
        "Hello World  ",
        "  Hello World",
        " \n Hello World \n ",
        " \n\n Hello World \n\n ",
    ]
    out = [
        " _ _h_ə_l_ˈ_oʊ_ _w_ˈ_ɜː_l_d_ _ ",
        "h_ə_l_ˈ_oʊ_ _w_ˈ_ɜː_l_d_ _ ",
        " _ _h_ə_l_ˈ_oʊ_ _w_ˈ_ɜː_l_d",
        " _\n_ _h_ə_l_ˈ_oʊ_ _w_ˈ_ɜː_l_d_ _\n_ ",
        " _\n\n_ _h_ə_l_ˈ_oʊ_ _w_ˈ_ɜː_l_d_ _\n\n_ ",
    ]
    assert grapheme_to_phoneme(in_, separator="_") == out


def test_grapheme_to_phoneme__white_space():
    """Test `grapheme_to_phoneme` preserves white spaces, SOMETIMES."""
    # NOTE: Test a number of string literals, see: https://docs.python.org/2.0/ref/strings.html
    # TODO: Which of these punctuation marks need to be preserved?
    in_ = " \n test \t test \r test \v test \f test \a test \b "
    out = " _\n_ _t_ˈ_ɛ_s_t_ _t_ˈ_ɛ_s_t_ _t_ˈ_ɛ_s_t_ _t_ˈ_ɛ_s_t_ _t_ˈ_ɛ_s_t_ _t_ˈ_ɛ_s_t_ "
    assert grapheme_to_phoneme([in_], separator="_") == [out]

    # TODO: Multiple white-spaces are not preserved.
    in_ = "résumé résumé  résumé   résumé"
    out = "ɹ|ˈ|ɛ|z|uː|m|ˌ|eɪ| |ɹ|ˈ|ɛ|z|uː|m|ˌ|eɪ| |ɹ|ˈ|ɛ|z|uː|m|ˌ|eɪ| |ɹ|ˈ|ɛ|z|uː|m|ˌ|eɪ"
    assert grapheme_to_phoneme([in_], separator="|") == [out]


def test_grapheme_to_phoneme__service_separator():
    """Test `grapheme_to_phoneme` works when `separator == service_separator`."""
    assert grapheme_to_phoneme(["Hello World"], separator="_") == ["h_ə_l_ˈ_oʊ_ _w_ˈ_ɜː_l_d"]


def test_grapheme_to_phoneme__unique_separator():
    """Test `grapheme_to_phoneme` errors if `separator` is not unique."""
    with pytest.raises(AssertionError):
        grapheme_to_phoneme(["Hello World"], separator="ə")


@mock.patch("lib.text.utils.logger.warning")
def test_grapheme_to_phoneme__language_switching(mock_warning):
    """Test `grapheme_to_phoneme` logs a warning if the language is switched."""
    assert grapheme_to_phoneme(["mon dieu"], separator="|") == ["m|ˈ|ɑː|n| |d|j|ˈ|ø"]
    assert mock_warning.called == 1


def test_grapheme_to_phoneme__special_bash_character():
    """Test `grapheme_to_phoneme` handles double quotes, a bash special character."""
    in_ = ['"It is commonly argued that the notion of']
    out = ["ɪ_t_ _ɪ_z_ _k_ˈ_ɑː_m_ə_n_l_i_ _ˈ_ɑːɹ_ɡ_j_uː_d_ _ð_æ_t_ð_ə_ _n_ˈ_oʊ_ʃ_ə_n_ _ʌ_v"]
    assert grapheme_to_phoneme(in_, separator="_") == out


def test_grapheme_to_phoneme__long_number():
    """Test `grapheme_to_phoneme` is UNABLE to handle long numbers.

    NOTE: eSpeak stops before outputing "7169399375105820974944592". Feel free to test this, like
    so: `espeak --ipa=3 -q -ven-us 3.141592653589793238462643383279502884197`.

    Learn more here: https://github.com/wellsaid-labs/Text-to-Speech/issues/299
    """
    in_ = "3.141592653589793238462643383279502884197169399375105820974944592"
    assert grapheme_to_phoneme([in_], separator="|") == [
        "θ|ɹ|ˈ|iː| |p|ɔɪ|n|t| |w|ˈ|ʌ|n| |f|ˈ|oːɹ| |w|ˈ|ʌ|n| |f|ˈ|aɪ|v| |n|ˈ|aɪ|n| |t|ˈ|uː| "
        "|s|ˈ|ɪ|k|s| |f|ˈ|aɪ|v| |θ|ɹ|ˈ|iː| |f|ˈ|aɪ|v| |ˈ|eɪ|t| |n|ˈ|aɪ|n| |s|ˈ|ɛ|v|ə|n| "
        "|n|ˈ|aɪ|n| |θ|ɹ|ˈ|iː| |t|ˈ|uː| |θ|ɹ|ˈ|iː| |ˈ|eɪ|t| |f|ˈ|oːɹ| |s|ˈ|ɪ|k|s| |t|ˈ|uː| "
        "|s|ˈ|ɪ|k|s| |f|ˈ|oːɹ| |θ|ɹ|ˈ|iː| |θ|ɹ|ˈ|iː| |ˈ|eɪ|t| |θ|ɹ|ˈ|iː| |t|ˈ|uː| "
        "|s|ˈ|ɛ|v|ə|n| |n|ˈ|aɪ|n| |f|ˈ|aɪ|v| |z|ˈ|iə|ɹ|oʊ| |t|ˈ|uː| |ˈ|eɪ|t| |ˈ|eɪ|t| "
        "|f|ˈ|oːɹ| |w|ˈ|ʌ|n| |n|ˈ|aɪ"
    ]


def test_grapheme_to_phoneme__empty():
    """Test `grapheme_to_phoneme` against an empty list."""
    assert grapheme_to_phoneme([]) == []


def test_grapheme_to_phoneme__regressions():
    """Test `grapheme_to_phoneme` against many real world examples."""
    inputs = [
        "and Trot fed it a handful of fresh blue clover and smoothed and petted it until the lamb "
        "was eager to follow her wherever she might go.",
        "The habits of mind that characterize a person strongly disposed toward critical thinking "
        "include a desire to follow reason and evidence wherever they may lead,",
        "The habits of mind that characterize a person strongly disposed toward critical thinking "
        "include a desire to follow reason and evidence wherever they may lead, a systematic",
        "But wherever they fought - in North Africa or the South Pacific or Western Europe -- the "
        "infantry bore the brunt of the fighting on the ground -- and seven out of ten suffered "
        "casualties.",
        "I lay eggs wherever I happen to be, said the hen, ruffling her feathers and then shaking "
        "them into place.",
        "scurrying for major stories whenever and wherever they could be found.",
        "actions .Sameer M Babu is a professor who wrote an article about classroom climate and "
        "social intelligence.",
        "copy by",
        "For example,",
        "(or sometimes being eliminated",
        """of 5 stages:
 (i) preparation,
 (ii) incubation,
 (iii) intimation,
 (iv) illumination""",
        "I ha thought till my brains ached,-Beli me, John, I have. An I say again, theres no help "
        "for us but having faith i the Union. Theyll win the day, see if they dunnot!",
        """the football play of the decade, or the concert of a lifetime.

They'll spend one-point-five billion dollars on twenty-eight thousand """
        "events ranging from Broadway to Super Bowls,",
        """Fortunately, Club Med has given us an antidote.

...The Club Med Vacation Village. Where all those prime disturbers of the peace like """
        "telephones, clocks and newspapers are gone.",
        """or fought over this ground.
--For this community, an ordeal that started with offense, uncertainty, and outrage, ended """
        "amidst horror, poverty,",
        """a band collar shirt buttoned tight around the throat and a dark business jacket.

He posed the couple, board-stiff in front of a plain house.

The man,""",
    ]
    outputs = [
        "æ_n_d_ _t_ɹ_ˈ_ɑː_t_ _f_ˈ_ɛ_d_ _ɪ_t_ _ɐ_ _h_ˈ_æ_n_d_f_əl_ _ʌ_v_ _f_ɹ_ˈ_ɛ_ʃ_ _b_l_ˈ_uː_ _"
        "k_l_ˈ_oʊ_v_ɚ_ _æ_n_d_ _s_m_ˈ_uː_ð_d_ _æ_n_d_ _p_ˈ_ɛ_ɾ_ᵻ_d_ _ɪ_t_ _ʌ_n_t_ˈ_ɪ_l_ _ð_ə_ _"
        "l_ˈ_æ_m_ _w_ʌ_z_ _ˈ_iː_ɡ_ɚ_ _t_ə_ _f_ˈ_ɑː_l_oʊ_ _h_ɜː_ _w_ɛɹ_ɹ_ˈ_ɛ_v_ɚ_ _ʃ_iː_ _m_ˌ_aɪ_t_ "
        "_ɡ_ˈ_oʊ",
        "ð_ə_ _h_ˈ_æ_b_ɪ_t_s_ _ʌ_v_ _m_ˈ_aɪ_n_d_ _ð_æ_t_ _k_ˈ_æ_ɹ_ɪ_k_t_ɚ_ɹ_ˌ_aɪ_z_ _ɐ_ _"
        "p_ˈ_ɜː_s_ə_n_ _s_t_ɹ_ˈ_ɔ_ŋ_l_i_ _d_ɪ_s_p_ˈ_oʊ_z_d_ _t_ʊ_w_ˈ_ɔːɹ_d_ _"
        "k_ɹ_ˈ_ɪ_ɾ_ɪ_k_əl_ _θ_ˈ_ɪ_ŋ_k_ɪ_ŋ_ _ɪ_n_k_l_ˈ_uː_d_ _ɐ_ _d_ɪ_z_ˈ_aɪɚ_ _t_ə_ _"
        "f_ˈ_ɑː_l_oʊ_ _ɹ_ˈ_iː_z_ə_n_ _æ_n_d_ _ˈ_ɛ_v_ɪ_d_ə_n_s_ _w_ɛɹ_ɹ_ˈ_ɛ_v_ɚ_ _ð_eɪ_ _"
        "m_ˈ_eɪ_ _l_ˈ_iː_d",
        "ð_ə_ _h_ˈ_æ_b_ɪ_t_s_ _ʌ_v_ _m_ˈ_aɪ_n_d_ _ð_æ_t_ _k_ˈ_æ_ɹ_ɪ_k_t_ɚ_ɹ_ˌ_aɪ_z_ _ɐ_ _"
        "p_ˈ_ɜː_s_ə_n_ _s_t_ɹ_ˈ_ɔ_ŋ_l_i_ _d_ɪ_s_p_ˈ_oʊ_z_d_ _t_ʊ_w_ˈ_ɔːɹ_d_ _"
        "k_ɹ_ˈ_ɪ_ɾ_ɪ_k_əl_ _θ_ˈ_ɪ_ŋ_k_ɪ_ŋ_ _ɪ_n_k_l_ˈ_uː_d_ _ɐ_ _d_ɪ_z_ˈ_aɪɚ_ _t_ə_ _"
        "f_ˈ_ɑː_l_oʊ_ _ɹ_ˈ_iː_z_ə_n_ _æ_n_d_ _ˈ_ɛ_v_ɪ_d_ə_n_s_ _w_ɛɹ_ɹ_ˈ_ɛ_v_ɚ_ _"
        "ð_eɪ_ _m_ˈ_eɪ_ _l_ˈ_iː_d_ _ɐ_ _s_ˌ_ɪ_s_t_ə_m_ˈ_æ_ɾ_ɪ_k",
        "b_ˌ_ʌ_t_ _w_ɛɹ_ɹ_ˈ_ɛ_v_ɚ_ _ð_eɪ_ _f_ˈ_ɔː_t_ _ɪ_n_ _n_ˈ_ɔːɹ_θ_ _ˈ_æ_f_ɹ_ɪ_k_ə_ _ɔːɹ_ _"
        "ð_ə_ _s_ˈ_aʊ_θ_ _p_ɐ_s_ˈ_ɪ_f_ɪ_k_ _ɔːɹ_ _w_ˈ_ɛ_s_t_ɚ_n_ _j_ˈ_ʊ_ɹ_ə_p_ _ð_ɪ_ _"
        "ˈ_ɪ_n_f_ə_n_t_ɹ_i_ _b_ˈ_oːɹ_ _ð_ə_ _b_ɹ_ˈ_ʌ_n_t_ _ʌ_v_ð_ə_ _f_ˈ_aɪ_ɾ_ɪ_ŋ_ _"
        "ɑː_n_ð_ə_ _ɡ_ɹ_ˈ_aʊ_n_d_ _æ_n_d_ _s_ˈ_ɛ_v_ə_n_ _ˌ_aʊ_ɾ_ə_v_ _t_ˈ_ɛ_n_ _s_ˈ_ʌ_f_ɚ_d_ _"
        "k_ˈ_æ_ʒ_uː_əl_ɾ_ɪ_z",
        "aɪ_ _l_ˈ_eɪ_ _ˈ_ɛ_ɡ_z_ _w_ɛɹ_ɹ_ˈ_ɛ_v_ɚ_ɹ_ _aɪ_ _h_ˈ_æ_p_ə_n_ _t_ə_ _b_ˈ_iː_ _s_ˈ_ɛ_d_ "
        "_ð_ə_ _h_ˈ_ɛ_n_ _ɹ_ˈ_ʌ_f_l_ɪ_ŋ_ _h_ɜː_ _f_ˈ_ɛ_ð_ɚ_z_ _æ_n_d_ _ð_ˈ_ɛ_n_ _ʃ_ˈ_eɪ_k_ɪ_ŋ_ _"
        "ð_ˌ_ɛ_m_ _ˌ_ɪ_n_t_ʊ_ _p_l_ˈ_eɪ_s",
        "s_k_ˈ_ɜː_ɹ_ɪ_ɪ_ŋ_ _f_ɔːɹ_ _m_ˈ_eɪ_dʒ_ɚ_ _s_t_ˈ_oː_ɹ_ɪ_z_ _w_ɛ_n_ˌ_ɛ_v_ɚ_ɹ_ _"
        "æ_n_d_ _w_ɛɹ_ɹ_ˈ_ɛ_v_ɚ_ _ð_eɪ_ _k_ʊ_d_ _b_iː_ _f_ˈ_aʊ_n_d",
        "ˈ_æ_k_ʃ_ə_n_z_ _d_ˈ_ɑː_t_ _s_æ_m_ˈ_ɪ_ɹ_ _ˈ_ɛ_m_ _b_ˈ_ɑː_b_uː_ _ɪ_z_ _ɐ_ _p_ɹ_ə_f_ˈ_"
        "ɛ_s_ɚ_ _h_ˌ_uː_ _ɹ_ˈ_oʊ_t_ _ɐ_n_ _ˈ_ɑːɹ_ɾ_ɪ_k_əl_ _ɐ_b_ˌ_aʊ_t_ _k_l_ˈ_æ_s_ɹ_uː_m_ _"
        "k_l_ˈ_aɪ_m_ə_t_ _æ_n_d_ _s_ˈ_oʊ_ʃ_əl_ _ɪ_n_t_ˈ_ɛ_l_ɪ_dʒ_ə_n_s",
        "k_ˈ_ɑː_p_i_ _b_ˈ_aɪ",
        "f_ɔː_ɹ_ _ɛ_ɡ_z_ˈ_æ_m_p_əl",
        "ɔːɹ_ _s_ˈ_ʌ_m_t_aɪ_m_z_ _b_ˌ_iː_ɪ_ŋ_ _ɪ_l_ˈ_ɪ_m_ᵻ_n_ˌ_eɪ_ɾ_ᵻ_d",
        """ʌ_v_ _f_ˈ_aɪ_v_ _s_t_ˈ_eɪ_dʒ_ᵻ_z_
_ _ˈ_aɪ_ _p_ɹ_ˌ_ɛ_p_ɚ_ɹ_ˈ_eɪ_ʃ_ə_n_
_ _ɹ_ˌ_oʊ_m_ə_n_ _t_ˈ_uː_ _ˌ_ɪ_n_k_j_uː_b_ˈ_eɪ_ʃ_ə_n_
_ _ɹ_ˌ_oʊ_m_ə_n_ _θ_ɹ_ˈ_iː_ _ˌ_ɪ_n_t_ɪ_m_ˈ_eɪ_ʃ_ə_n_
_ _ɹ_ˌ_oʊ_m_ə_n_ _f_ˈ_oːɹ_ _ɪ_l_ˌ_uː_m_ᵻ_n_ˈ_eɪ_ʃ_ə_n""",
        "aɪ_ _h_ˈ_ɑː_ _θ_ˈ_ɔː_t_ _t_ˈ_ɪ_l_ _m_aɪ_ _b_ɹ_ˈ_eɪ_n_z_ _ˈ_eɪ_k_t_b_ɪ_l_i_ _m_ˌ_iː_ "
        "_dʒ_ˈ_ɑː_n_ _aɪ_ _h_ˈ_æ_v_ _ɐ_n_ _aɪ_ _"
        "s_ˈ_eɪ_ _ɐ_ɡ_ˈ_ɛ_n_ _ð_ɚ_z_ _n_ˈ_oʊ_ _h_ˈ_ɛ_l_p_ _f_ɔː_ɹ_ _ˌ_ʌ_s_ _b_ˌ_ʌ_t_ _h_ˌ_æ_v_ɪ_ŋ_ "
        "_f_ˈ_eɪ_θ_ _ˈ_aɪ_ _ð_ə_ _"
        "j_ˈ_uː_n_iə_n_ _θ_ˈ_eɪ_l_ _w_ˈ_ɪ_n_ _ð_ə_ _d_ˈ_eɪ_ _s_ˈ_iː_ _ɪ_f_ _ð_eɪ_ _d_ˈ_ʌ_n_ɑː_t",
        "ð_ə_ _f_ˈ_ʊ_t_b_ɔː_l_ _p_l_ˈ_eɪ_ _ʌ_v_ð_ə_ _d_ˈ_ɛ_k_eɪ_d_ _ɔːɹ_ _ð_ə_ _k_ˈ_ɑː_n_s_ɜː_t_ _"
        "ə_v_ə_ _l_ˈ_aɪ_f_t_aɪ_m_\n_\n_"
        "ð_eɪ_l_ _s_p_ˈ_ɛ_n_d_ _w_ˈ_ʌ_n_p_ˈ_ɔɪ_n_t_f_ˈ_aɪ_v_ _b_ˈ_ɪ_l_iə_n_ _d_ˈ_ɑː_l_ɚ_z_ _"
        "ˌ_ɑː_n_ _t_w_ˈ_ɛ_n_t_i_ˈ_eɪ_t_ _θ_ˈ_aʊ_z_ə_n_d_ _ɪ_v_ˈ_ɛ_n_t_s_ _ɹ_ˈ_eɪ_n_dʒ_ɪ_ŋ_ _"
        "f_ɹ_ʌ_m_ _b_ɹ_ˈ_ɔː_d_w_eɪ_ _t_ə_ _s_ˈ_uː_p_ɚ_ _b_ˈ_oʊ_l_z",
        "f_ˈ_ɔːɹ_tʃ_ə_n_ə_t_l_i_ _k_l_ˈ_ʌ_b_ _m_ˈ_ɛ_d_ _h_ɐ_z_ _ɡ_ˈ_ɪ_v_ə_n_ _ˌ_ʌ_s_ _ɐ_n_ "
        "_ˈ_æ_n_t_ɪ_d_ˌ_oʊ_t_\n_\n_"
        "ð_ə_ _k_l_ˈ_ʌ_b_ _m_ˈ_ɛ_d_ _v_eɪ_k_ˈ_eɪ_ʃ_ə_n_ _v_ˈ_ɪ_l_ɪ_dʒ_ _w_ˌ_ɛ_ɹ_ _ˈ_ɔː_l_ _ð_oʊ_z_ "
        "_p_ɹ_ˈ_aɪ_m_ _"
        "d_ɪ_s_t_ˈ_ɜː_b_ɚ_z_ _ʌ_v_ð_ə_ _p_ˈ_iː_s_ _l_ˈ_aɪ_k_ _t_ˈ_ɛ_l_ɪ_f_ˌ_oʊ_n_z_ _k_l_ˈ_ɑː_k_s_"
        " _æ_n_d_ _n_ˈ_uː_z_p_eɪ_p_ɚ_z_ _ɑːɹ_ _ɡ_ˈ_ɔ_n",
        "ɔːɹ_ _f_ˈ_ɔː_t_ _ˌ_oʊ_v_ɚ_ _ð_ɪ_s_ _ɡ_ɹ_ˈ_aʊ_n_d_\n_"
        "f_ɔːɹ_ _ð_ɪ_s_ _k_ə_m_j_ˈ_uː_n_ɪ_ɾ_i_ _ɐ_n_ _ɔːɹ_d_ˈ_iə_l_ _ð_æ_t_ _s_t_ˈ_ɑːɹ_ɾ_ᵻ_d_ "
        "_w_ɪ_ð_ _ə_f_ˈ_ɛ_n_s_ _ʌ_n_s_ˈ_ɜː_t_ə_n_t_i_ _æ_n_d_ _ˈ_aʊ_t_ɹ_eɪ_dʒ_ _ˈ_ɛ_n_d_ᵻ_d_ "
        "_ɐ_m_ˈ_ɪ_d_s_t_ _h_ˈ_ɔː_ɹ_ɚ_ _p_ˈ_ɑː_v_ɚ_ɾ_i",
        "ɐ_ _b_ˈ_æ_n_d_ _k_ˈ_ɑː_l_ɚ_ _ʃ_ˈ_ɜː_t_ _b_ˈ_ʌ_ʔ_n̩_d_ _t_ˈ_aɪ_t_ _ɐ_ɹ_ˈ_aʊ_n_d_ _ð_ə_ "
        "_θ_ɹ_ˈ_oʊ_t_ _æ_n_d_ _ɐ_ _d_ˈ_ɑːɹ_k_ _b_ˈ_ɪ_z_n_ə_s_ _dʒ_ˈ_æ_k_ɪ_t_\n_\n_"
        "h_iː_ _p_ˈ_oʊ_z_d_ _ð_ə_ _k_ˈ_ʌ_p_əl_ _b_ˈ_oːɹ_d_s_t_ˈ_ɪ_f_ _ɪ_n_ _f_ɹ_ˈ_ʌ_n_t_ _ə_v_ə_ "
        "_p_l_ˈ_eɪ_n_ _h_ˈ_aʊ_s_\n_\n_ð_ə_ _m_ˈ_æ_n",
    ]
    for in_, out in zip(inputs, outputs):
        for i, o in zip(grapheme_to_phoneme(in_.split("\n"), separator="_"), out.split("\n")):
            assert i.strip("_") == o.strip("_")


def test_load_cmudict_syl():
    """Test `lib.text.utils.load_cmudict_syl` loads the dictionary."""
    dictionary = lib.text.utils.load_cmudict_syl()
    arpabet: typing.Set[lib.text.utils.ARPAbet] = set()
    characters = set()
    for word, pronunciations in dictionary.items():
        for pronunciation in pronunciations:
            assert len(word) == len(word.strip())
            assert len(word) > 0
            arpabet.update([code for syllable in pronunciation for code in syllable])
            characters.update(list(word))
    assert arpabet == set(get_args(lib.text.utils.ARPAbet))
    assert characters == set(list(string.ascii_uppercase) + ["'"])


def _check_pronunciation(word: str, expected: typing.Optional[str] = None):
    result = lib.text.utils.get_pronunciation(word, lib.text.utils.load_cmudict_syl())
    if expected is None:
        assert result is expected
    else:
        assert result == tuple(tuple(s.split()) for s in expected.split(" - "))


def test_get_pronunciation():
    """Test `lib.text.utils.get_pronunciation` on basic scenarios."""
    expectations = {
        "zebra": (("Z", "IY1"), ("B", "R", "AH0")),
        "motorcycle": (("M", "OW1"), ("T", "ER0"), ("S", "AY2"), ("K", "AH0", "L")),
        "suspicious": (("S", "AH0"), ("S", "P", "IH1"), ("SH", "AH0", "S")),
    }
    for word, expected in expectations.items():
        pronunication = lib.text.utils.get_pronunciation(word, lib.text.utils.load_cmudict_syl())
        assert expected == pronunication


def test_get_pronunciation__out_of_vocabulary():
    """Test `lib.text.utils.get_pronunciation` doesn't handle words outside it's vocabulary."""
    _check_pronunciation("abcdefg", expected=None)
    _check_pronunciation(" ", expected=None)
    _check_pronunciation("\t", expected=None)


def test_get_pronunciation__apostrophes():
    """Test `lib.text.utils.get_pronunciation` handles apostrophes at the end and beginning of
    a word."""
    _check_pronunciation("accountants'", expected="AH0 - K AW1 N - T AH0 N T S")
    _check_pronunciation("'bout", expected=None)


def test_get_pronunciation__variations():
    """Test `lib.text.utils.get_pronunciation` doesn't return if the pronunciation is ambigious."""
    # NOTE: Base case with no variations to choose from.
    _check_pronunciation("fly", expected="F L AY1")

    # NOTE: Multiple variations that can only be disambiguated with part-of-speech.
    _check_pronunciation("read", expected=None)
    _check_pronunciation("beloved", expected=None)

    # NOTE: Abbreviations that are sometimes expanded during voice-over are not disambiguated.
    _check_pronunciation("feb", expected=None)


def test_get_pronunciation__non_standard_words():
    """Test `lib.text.utils.get_pronunciation` returns `None` given non-standard words."""
    _check_pronunciation("I B M", expected=None)
    _check_pronunciation("I.B.M.", expected=None)
    _check_pronunciation("able-bodied", expected=None)
    _check_pronunciation("ABC123", expected=None)


def test_respell():
    """Test `lib.text.utils.respell` on basic scenarios."""
    expectations = {
        "zebra": "ZEE-bruh",
        "motorcycle": "MOH-tur-sy-kuhl",  # NOTE: Secondary is lowercase if primary is uppercase.
        "suspicious": "suh-SPIH-shuhs",
        "maui": "MOW-ee",
        "cobalt": "KOH-bawlt",  # NOTE: Wikipedia recommends "KOH-bolt"
        "father": "FAH-dhur",
        "farther": "FAR-dhur",  # NOTE: Wikipedia recommends "FAR-dhər", testing AA R : ahr -> ar
        "ceres": "SEE-reez",  # NOTE: Wikipedia recommends "SEER-eez"
        "algorithm": "AL-gur-ih-dhuhm",  # NOTE: Wikipedia recommends "AL-gə-ridh-əm"
        "pan": "PAN",
        "machine": "muh-SHEEN",  # NOTE: Wikipedia recommends "mə-SHEEN",
        "blank": "BLAYNK",  # Testing AE NG K : angk -> aynk
        "blanketed": "BLAYNG-kuh-tihd",  # Testing AE NG : ang -> ayng
        "ink": "IHNK",  # Testing NG K : ngk -> nk
        "millionaire": "MIH-lyuh-NERR",  # Testing EH R : ehr -> err
        "engineer": "EHN-juh-NEER",  # Testing IH R : ihr -> eer
        "mirror": "MEE-rur",  # Testing IH - R : ih-r -> ee-r
        "storyboard": "STOH-ree-bord",  # Testing AO - r : AW-r -> OH-r and AO R : awr -> or
    }
    for word, pronunciation in expectations.items():
        assert pronunciation == lib.text.utils.respell(word, lib.text.utils.load_cmudict_syl())


def test_respell__vowels():
    """Test `lib.text.utils.respell` on various vowels."""
    vowel_expectations = {
        "bat": "BAT",
        "father": "FAH-dhur",
        "oddball": "AHD-bawl",
        "straighten": "STRAY-tuhn",
        "happy": "HA-pee",
        "prestige": "preh-STEEZH",
        "about": "uh-BOWT",
        "letter": "LEH-tur",
        "historic": "hih-STOH-rihk",
        "boat": "BOHT",
        "boot": "BOOT",
        "flower": "FLOW-ur",
        "joy": "JOY",
        "jump": "JUHMP",
        "nook": "NUUK",
        "site": "SYT",
    }
    for word, pronunciation in vowel_expectations.items():
        assert pronunciation == lib.text.utils.respell(word, lib.text.utils.load_cmudict_syl())


def test_respell__consonant():
    """Test `lib.text.utils.respell` on various consonant."""
    consonant_expectations = {
        "bunk": "BUHNK",
        "dusters": "DUH-sturz",
        "although": "AWL-DHOH",
        "firstly": "FURST-lee",
        "global": "GLOH-buhl",
        "horse": "HORS",
        "jealous": "JEH-luhs",
        "kite": "KYT",
        "believe": "bih-LEEV",
        "flammable": "FLA-muh-buhl",
        "friend": "FREHND",
        "singing": "SIH-ngihng",
        "people": "PEE-puhl",
        "rascal": "RAS-kuhl",
        "slice": "SLYS",
        "shy": "SHY",
        "turtle": "TUR-tuhl",
        "nature": "NAY-chur",
        "thinks": "THIHNKS",
        "save": "SAYV",
        "win": "WIHN",
        "yesteryear": "YEH-stur-yeer",
        "please": "PLEEZ",
        "measure": "MEH-zhur",
    }
    for word, pronunciation in consonant_expectations.items():
        assert pronunciation == lib.text.utils.respell(word, lib.text.utils.load_cmudict_syl())


def test_respell_initialism():
    """Test `lib.text.utils.respell_initialism` on initialisms and acronyms."""
    expectations = {
        "FBI": "ehf-bee-Y",
        "RSVP": "ar-ehs-vee-PEE",
        "FAQ": "ehf-ay-KYOO",
        "UN": "yoo-EHN",
        "LGBTQIA": "ehl-jee-bee-tee-kyoo-y-AY",
        "FSW": "ehf-ehs-DUH-buhl-yoo",  # Test final W does not have all capitalized syllables
    }
    for initialism, pronunciation in expectations.items():
        assert pronunciation == lib.text.utils.respell_initialism(initialism)


def test_respellings():
    """Test to ensure `RESPELLINGS` covers all of `ARPAbet`."""
    assert all(_remove_arpabet_markings(a) in RESPELLINGS for a in get_args(lib.text.utils.ARPAbet))


def test_respellings_alphabet():
    """Test to ensure `RESPELLING_ALPHABET` covers the entire alphabet."""
    assert all(a in RESPELLING_ALPHABET for a in string.ascii_uppercase)


def test_natural_keys():
    """Test `lib.text.utils.natural_keys` sorts naturally."""
    list_ = ["name 0", "name 1", "name 10", "name 11"]
    assert sorted(list_, key=lib.text.utils.natural_keys) == list_


def test_strip():
    """Test `lib.text.utils.strip` handles various white space scenarios."""
    assert lib.text.utils.strip("  Hello World  ") == ("Hello World", "  ", "  ")
    assert lib.text.utils.strip("Hello World  ") == ("Hello World", "", "  ")
    assert lib.text.utils.strip("  Hello World") == ("Hello World", "  ", "")
    assert lib.text.utils.strip(" \n Hello World \n ") == ("Hello World", " \n ", " \n ")
    assert lib.text.utils.strip(" \n\n Hello World \n\n ") == ("Hello World", " \n\n ", " \n\n ")


def test_normalize_vo_script():
    """Test `lib.text.utils.normalize_vo_script` handles all characters from 0 - 128."""
    # fmt: off
    assert list(normalize_vo_script(chr(i), frozenset(), strip=False) for i in range(0, 128)) == [
        "", "", "", "", "", "", "", "", "", "  ", "\n", "", "\n", "\n", "", "", "", "", "", "", "",
        "", "", "", "", "", "", "", "", "", "", "", " ", "!", '"', "#", "$", "%", "&", "'", "(",
        ")", "*", "+", ",", "-", ".", "/", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", ":",
        ";", "<", "=", ">", "?", "@", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L",
        "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "[", "\\", "]", "^",
        "_", "`", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p",
        "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "{", "|", "}", "~", ""
    ]
    # fmt: on

    # Cover whitespace normalization, common whitespace issues
    assert normalize_vo_script("\r\n", frozenset(), strip=False) == "\n"
    assert normalize_vo_script("\f", frozenset(), strip=False) == "\n"
    assert normalize_vo_script("\thello\t", frozenset(), strip=False) == "  hello  "

    # Cover guillemet and quotation normalization
    assert all(
        [
            p == '"Wir gehen am Dienstag."'
            for p in [
                normalize_vo_script("»Wir gehen am Dienstag.«", frozenset(), strip=False),
                normalize_vo_script("„Wir gehen am Dienstag.”", frozenset(), strip=False),
            ]
        ]
    )
    # TODO: Need to Fix. Currently failing because 12½ normalizes to 121/2 instead of 12 1/2.
    # assert (
    #     normalize_vo_script("It's 12½ km² in area.", frozenset(), strip=False)
    #     == "It's 12 1/2 km2 in area."
    # )
    assert (
        normalize_vo_script("‹Wir gehen am Dienstag.›", frozenset(), strip=False)
        == "'Wir gehen am Dienstag.'"
    )


# fmt: off
NON_ASCII_CHAR = frozenset([
    'Á', 'Ñ', '¡', 'Í', 'á', 'û', 'Ç', 'É', 'Ô', 'ß', 'ó', 'è', 'ú', 'Ì', 'Ù', 'Ó', 'ô', 'ù', 'ã',
    'Ú', 'õ', 'ï', 'â', 'Ï', 'ò', 'À', 'é', 'à', 'ö', 'ü', 'ì', 'Ü', 'ç', 'Û', 'È', 'ë', 'ä', 'Ä',
    'Ö', 'Â', 'Ò', 'Î', 'Õ', 'Ê', 'î', '¿', 'Ë', 'ñ', 'ê', 'Ã', 'í',
])
# fmt: on


def test_normalize_vo_script__non_ascii():
    """Test `lib.text.utils.normalize_vo_script` allows through select non-ascii characters."""
    for char in NON_ASCII_CHAR:
        assert lib.text.utils.normalize_vo_script(char, non_ascii=NON_ASCII_CHAR) == char


def test_is_normalized_vo_script():
    """Test `lib.text.utils.is_normalized_vo_script` handles all characters from 0 - 128."""
    assert all(
        is_normalized_vo_script(normalize_vo_script(chr(i), frozenset(), strip=False), frozenset())
        for i in range(0, 128)
    )


def test_is_normalized_vo_script__non_ascii():
    """Test `lib.text.utils.is_normalized_vo_script` handles non-ascii characters."""
    for char in NON_ASCII_CHAR:
        assert lib.text.utils.is_normalized_vo_script(char, non_ascii=NON_ASCII_CHAR)


def test_is_normalized_vo_script__unnormalized():
    """Test `lib.text.utils.is_normalized_vo_script` fails for unnormalized characters."""
    # fmt: off
    assert [(chr(i), is_normalized_vo_script(chr(i), frozenset())) for i in range(0, 128)] == [
        ("\x00", False), ("\x01", False), ("\x02", False), ("\x03", False), ("\x04", False),
        ("\x05", False), ("\x06", False), ("\x07", False), ("\x08", False), ("\t", False),
        ("\n", True), ("\x0b", False), ("\x0c", False), ("\r", False), ("\x0e", False),
        ("\x0f", False), ("\x10", False), ("\x11", False), ("\x12", False), ("\x13", False),
        ("\x14", False), ("\x15", False), ("\x16", False), ("\x17", False), ("\x18", False),
        ("\x19", False), ("\x1a", False), ("\x1b", False), ("\x1c", False), ("\x1d", False),
        ("\x1e", False), ("\x1f", False), (" ", True), ("!", True), ('"', True), ("#", True),
        ("$", True), ("%", True), ("&", True), ("'", True), ("(", True), (")", True), ("*", True),
        ("+", True), (",", True), ("-", True), (".", True), ("/", True), ("0", True), ("1", True),
        ("2", True), ("3", True), ("4", True), ("5", True), ("6", True), ("7", True), ("8", True),
        ("9", True), (":", True), (";", True), ("<", True), ("=", True), (">", True), ("?", True),
        ("@", True), ("A", True), ("B", True), ("C", True), ("D", True), ("E", True), ("F", True),
        ("G", True), ("H", True), ("I", True), ("J", True), ("K", True), ("L", True), ("M", True),
        ("N", True), ("O", True), ("P", True), ("Q", True), ("R", True), ("S", True), ("T", True),
        ("U", True), ("V", True), ("W", True), ("X", True), ("Y", True), ("Z", True), ("[", True),
        ("\\", True), ("]", True), ("^", True), ("_", True), ("`", True), ("a", True), ("b", True),
        ("c", True), ("d", True), ("e", True), ("f", True), ("g", True), ("h", True), ("i", True),
        ("j", True), ("k", True), ("l", True), ("m", True), ("n", True), ("o", True), ("p", True),
        ("q", True), ("r", True), ("s", True), ("t", True), ("u", True), ("v", True), ("w", True),
        ("x", True), ("y", True), ("z", True), ("{", True), ("|", True), ("}", True), ("~", True),
        ("\x7f", False),
    ]
    # fmt: on


def test_is_voiced():
    """Test `lib.text.utils.is_voiced` handles all characters and an empty string."""
    assert lib.text.utils.is_voiced("123", frozenset())
    assert lib.text.utils.is_voiced("abc", frozenset())
    assert lib.text.utils.is_voiced("ABC", frozenset())
    for char in "@#$%&+=*".split():
        assert lib.text.utils.is_voiced(char, frozenset())
    assert not lib.text.utils.is_voiced("!^()_{[}]:;\"'<>?/~`|\\", frozenset())
    assert not lib.text.utils.is_voiced("", frozenset())


def test_is_voiced__non_ascii():
    """Test `lib.text.utils.is_voiced` handles non-ascii characters."""
    for char in NON_ASCII_CHAR:
        assert lib.text.utils.is_voiced(char, non_ascii=NON_ASCII_CHAR)


def test_has_digit():
    """Test `lib.text.utils.has_digit` handles basic cases."""
    assert lib.text.utils.has_digit("123")
    assert lib.text.utils.has_digit("123abc")
    assert lib.text.utils.has_digit("1")
    assert not lib.text.utils.has_digit("abc")
    assert not lib.text.utils.has_digit("")


def test_get_spoken_chars():
    """Test `get_spoken_chars` removes marks, spaces and casing."""
    pattern = re.compile(r"[^\w\s]")
    assert lib.text.utils.get_spoken_chars("123 abc !.?", pattern) == "123abc"
    assert lib.text.utils.get_spoken_chars("Hello. You've", pattern) == "helloyouve"
    assert lib.text.utils.get_spoken_chars("Hello. \n\fYou've", pattern) == "helloyouve"


def test_add_space_between_sentences():
    """Test `lib.text.utils.add_space_between_sentences` adds a space between sentences."""
    nlp = lib.text.utils.load_en_core_web_sm()
    script = (
        "Business was involved in slavery, colonialism, and the cold war.The term "
        "'business ethics' came into common use in the United States in the early 1970s."
    )
    assert lib.text.utils.add_space_between_sentences(nlp(script)) == (
        "Business was involved in slavery, colonialism, and the cold war. The term "
        "'business ethics' came into common use in the United States in the early 1970s."
    )
    script = (
        "Mix and match the textured shades for a funky effect.Hang on "
        "to these fuzzy hangers from Domis."
    )
    assert lib.text.utils.add_space_between_sentences(nlp(script)) == (
        "Mix and match the textured shades for a funky effect. Hang on "
        "to these fuzzy hangers from Domis."
    )


def test_add_space_between_sentences__new_lines():
    """Test `lib.text.utils.add_space_between_sentences` adds a space between sentences while handling
    newlines."""
    nlp = lib.text.utils.load_en_core_web_sm()
    script = """
    The neuroscience of creativity looks at the operation of the brain during creative behaviour.
    It has been addressed in the article "Creative Innovation: Possible Brain Mechanisms."
    The authors write that "creative innovation might require coactivation and communication
    between regions of the brain that ordinarily are not strongly connected." Highly creative
    people who excel at creative innovation tend to differ from others in three ways:

    they have a high level of specialized knowledge,
    they are capable of divergent thinking mediated by the frontal lobe.
    and they are able to modulate neurotransmitters such as norepinephrine in their
    frontal lobe.Thus, the frontal lobe appears to be the part of the cortex that is most important
    for creativity.
    """
    expected = """
    The neuroscience of creativity looks at the operation of the brain during creative behaviour.
    It has been addressed in the article "Creative Innovation: Possible Brain Mechanisms."
    The authors write that "creative innovation might require coactivation and communication
    between regions of the brain that ordinarily are not strongly connected." Highly creative
    people who excel at creative innovation tend to differ from others in three ways:

    they have a high level of specialized knowledge,
    they are capable of divergent thinking mediated by the frontal lobe.
    and they are able to modulate neurotransmitters such as norepinephrine in their
    frontal lobe. Thus, the frontal lobe appears to be the part of the cortex that is most important
    for creativity.
    """
    assert lib.text.utils.add_space_between_sentences(nlp(script)) == expected


def test_add_space_between_sentences__one_word():
    """Test `lib.text.utils.add_space_between_sentences` handles one word."""
    nlp = lib.text.utils.load_en_core_web_sm()
    assert lib.text.utils.add_space_between_sentences(nlp("Hi")) == "Hi"
    assert lib.text.utils.add_space_between_sentences(nlp("Hi  ")) == "Hi  "
    assert lib.text.utils.add_space_between_sentences(nlp("Hi.  ")) == "Hi.  "


def test_add_space_between_sentences__regression():
    """Test `lib.text.utils.add_space_between_sentences` handles these regression tests on Hilary's
    data."""
    nlp = lib.text.utils.load_en_core_web_sm()
    fixed = [
        (
            "The business' actions and decisions should be primarily ethical before it happens to "
            'become an ethical or even legal issue. "In the case of the government, community, and '
            "society what was merely an ethical issue can become a legal debate and eventually "
            'law."'
        ),
        (
            "In Sharia law, followed by many Muslims, banking specifically prohibits charging "
            "interest on loans. Traditional Confucian thought discourages profit-seeking. "
            'Christianity offers the Golden Rule command, "Therefore all things whatsoever ye '
            "would that men should do to you, do ye even so to them: for this is the law "
            'and the prophets."'
        ),
        (
            "This joint focus highlights both the theoretical and practical importance of the "
            "relationship: researchers are interested not only if the constructs are related, but "
            "also how and why."
        ),
        (
            "This theorem allows one to use the assumption of a normal distribution when dealing "
            'with x-bar. "Sufficiently large" depends on the population\'s distribution and what '
            "range of x-bar is being considered; for practical the easiest approach may be to "
            "take a number of samples of a desired size and see if their means are normally "
            "distributed."
        ),
        (
            "In short, we can say that our company is...\n\n"
            "AE1.11: Administrative  Assistant: Your job plays a key role in the success of our "
            "company."
        ),
        (
            "Twice a year, day and night fall into balance, lasting for nearly equal lengths. "
            'Known as "equinoxes", Latin for "equal night" they occur in March and September, '
            "and along with solstices, mark the changing of seasons as earth travels around the "
            "sun."
        ),
        (
            "Figure Concepts, where participants were given simple drawings of objects and "
            "individuals and asked to find qualities or features that are common by two or more "
            "drawings; these were scored for uncommonness."
        ),
    ]
    for script in fixed:
        assert lib.text.utils.add_space_between_sentences(nlp(script)) == script


def _align_and_format(tokens, other, **kwargs):
    cost, alignment = lib.text.utils.align_tokens(tokens, other, **kwargs)
    return lib.text.utils.format_alignment(tokens, other, alignment)


def test_align_tokens__empty():
    """Test `lib.text.utils.align_tokens` aligns empty text correctly."""
    assert lib.text.utils.align_tokens("", "")[0] == 0
    assert lib.text.utils.align_tokens("a", "")[0] == 1
    assert lib.text.utils.align_tokens("", "a")[0] == 1
    assert lib.text.utils.align_tokens("abc", "")[0] == 3
    assert lib.text.utils.align_tokens("", "abc")[0] == 3
    assert lib.text.utils.align_tokens("", "abc", window_length=1)[0] == 3


def test_align_tokens__one_letter():
    """Test `lib.text.utils.align_tokens` aligns one letter correctly."""
    # Should just add "a" to the beginning.
    assert lib.text.utils.align_tokens("abc", "bc", window_length=1)[0] == 1
    assert lib.text.utils.align_tokens("abc", "bc", allow_substitution=lambda a, b: False)[0] == 1
    assert _align_and_format("abc", "bc") == (
        "a b c",
        "  b c",
    )

    # Should just add I to the beginning.
    assert lib.text.utils.align_tokens("islander", "slander")[0] == 1
    assert lib.text.utils.align_tokens("islander", "slander", window_length=1)[0] == 1
    assert _align_and_format("islander", "slander") == (
        "i s l a n d e r",
        "  s l a n d e r",
    )


def test_align_tokens__deletion():
    """Test `lib.text.utils.align_tokens` deletion."""
    # Should delete 4 letters FOOT at the beginning.
    assert lib.text.utils.align_tokens("football", "foot")[0] == 4


def test_align_tokens__substitution():
    """Test `lib.text.utils.align_tokens` substitution."""
    # Needs to substitute the first 5 chars: INTEN by EXECU
    assert lib.text.utils.align_tokens("intention", "execution")[0] == 5


def test_align_tokens__multi_operation_alignments():
    """Test `lib.text.utils.align_tokens` substitution, insertion, and deletion."""
    # Needs to substitute M by K, T by M and add an A to the end
    assert lib.text.utils.align_tokens("mart", "karma")[0] == 3

    # Needs to substitute K by M, M by T, and delete A from the end
    assert lib.text.utils.align_tokens("karma", "mart")[0] == 3

    # Substitute K by S, E by I and add a G at the end.
    assert lib.text.utils.align_tokens("kitten", "sitting")[0] == 3


def test_align_tokens__window_lengths():
    """Test `lib.text.utils.align_tokens` handles various window lengths."""
    assert lib.text.utils.align_tokens("ball", "football")[0] == 4
    assert lib.text.utils.align_tokens("ball", "football", window_length=1)[0] == 7
    assert _align_and_format("ball", "football", window_length=1) == (
        "b a l       l",
        "f o o t b a l",
    )
    assert lib.text.utils.align_tokens("ball", "football", window_length=2)[0] == 6
    assert _align_and_format("ball", "football", window_length=2) == (
        "b a         l l",
        "f o o t b a l l",
    )
    assert lib.text.utils.align_tokens("ball", "football", window_length=3)[0] == 4
    assert _align_and_format("ball", "football", window_length=3) == (
        "        b a l l",
        "f o o t b a l l",
    )


def test_align_tokens__word_subtitution():
    """Test `lib.text.utils.align_tokens` substitutes words."""
    assert lib.text.utils.align_tokens(["Hey", "There"], ["Hey", "There"])[0] == 0
    assert lib.text.utils.align_tokens(["Hey", "There"], ["Hi", "There"])[0] == 2
    assert lib.text.utils.align_tokens(["Hey", "There"], ["Hi", "The"])[0] == 4


def test_align_tokens__word_deletion():
    """Test `lib.text.utils.align_tokens` deletes words."""
    assert lib.text.utils.align_tokens(["Hey", "There", "You"], ["Hey", ",", "There"])[0] == 4
    assert _align_and_format(["Hey", "There", "You"], ["Hey", ",", "There"]) == (
        "Hey   There",
        "Hey , There",
    )

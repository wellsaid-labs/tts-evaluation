import string
import typing
from typing import get_args
from unittest import mock

import pytest

import lib
from lib.text import _line_grapheme_to_phoneme, _multiline_grapheme_to_phoneme, grapheme_to_phoneme


def test__line_grapheme_to_phoneme():
    """Test `_line_grapheme_to_phoneme` can handle some basic cases."""
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
    assert _line_grapheme_to_phoneme(in_, separator="_") == out


def test__multiline_grapheme_to_phoneme():
    """Test `_multiline_grapheme_to_phoneme` against basic cases."""
    in_ = [
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
    out = [
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
    assert _multiline_grapheme_to_phoneme(in_, separator="_") == out


def test__multiline_grapheme_to_phoneme__special_bash_character():
    """Test `_multiline_grapheme_to_phoneme` handles double quotes, a bash special character."""
    in_ = ['"It is commonly argued that the notion of']
    out = ["ɪ_t_ _ɪ_z_ _k_ˈ_ɑː_m_ə_n_l_i_ _ˈ_ɑːɹ_ɡ_j_uː_d_ _ð_æ_t_ð_ə_ _n_ˈ_oʊ_ʃ_ə_n_ _ʌ_v"]
    assert _multiline_grapheme_to_phoneme(in_, separator="_") == out


def test__multiline_grapheme_to_phoneme__stripping():
    """Test `_multiline_grapheme_to_phoneme` respects white spaces on the edges."""
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
        " _\n_\n_ _h_ə_l_ˈ_oʊ_ _w_ˈ_ɜː_l_d_ _\n_\n_ ",
    ]
    assert _multiline_grapheme_to_phoneme(in_, separator="_") == out


def test__multiline_grapheme_to_phoneme__service_separator():
    """Test `_multiline_grapheme_to_phoneme` works when `separator == service_separator`."""
    assert _multiline_grapheme_to_phoneme(["Hello World"], separator="_") == [
        "h_ə_l_ˈ_oʊ_ _w_ˈ_ɜː_l_d"
    ]


def test__multiline_grapheme_to_phoneme__unique_separator():
    """Test `_multiline_grapheme_to_phoneme` errors if `separator` is not unique."""
    with pytest.raises(AssertionError):
        _multiline_grapheme_to_phoneme(["Hello World"], separator="ə")


@mock.patch("lib.text.logger.warning")
def test__multiline_grapheme_to_phoneme__language_switching(mock_warning):
    """Test `_multiline_grapheme_to_phoneme` logs a warning if the language is switched."""
    assert _multiline_grapheme_to_phoneme(["mon dieu"], separator="|") == ["m|ˈ|ɑː|n| |d|j|ˈ|ø"]
    assert mock_warning.called == 1


def test__multiline_grapheme_to_phoneme__long_number():
    """Test `_multiline_grapheme_to_phoneme` is UNABLE to handle long numbers.

    NOTE: eSpeak stops before outputing "7169399375105820974944592". Feel free to test this, like
    so: `espeak --ipa=3 -q -ven-us 3.141592653589793238462643383279502884197`.

    Learn more here: https://github.com/wellsaid-labs/Text-to-Speech/issues/299
    """
    in_ = "3.141592653589793238462643383279502884197169399375105820974944592"
    assert _multiline_grapheme_to_phoneme([in_], separator="|") == [
        "θ|ɹ|ˈ|iː| |p|ɔɪ|n|t| |w|ˈ|ʌ|n| |f|ˈ|oːɹ| |w|ˈ|ʌ|n| |f|ˈ|aɪ|v| |n|ˈ|aɪ|n| |t|ˈ|uː| "
        "|s|ˈ|ɪ|k|s| |f|ˈ|aɪ|v| |θ|ɹ|ˈ|iː| |f|ˈ|aɪ|v| |ˈ|eɪ|t| |n|ˈ|aɪ|n| |s|ˈ|ɛ|v|ə|n| "
        "|n|ˈ|aɪ|n| |θ|ɹ|ˈ|iː| |t|ˈ|uː| |θ|ɹ|ˈ|iː| |ˈ|eɪ|t| |f|ˈ|oːɹ| |s|ˈ|ɪ|k|s| |t|ˈ|uː| "
        "|s|ˈ|ɪ|k|s| |f|ˈ|oːɹ| |θ|ɹ|ˈ|iː| |θ|ɹ|ˈ|iː| |ˈ|eɪ|t| |θ|ɹ|ˈ|iː| |t|ˈ|uː| "
        "|s|ˈ|ɛ|v|ə|n| |n|ˈ|aɪ|n| |f|ˈ|aɪ|v| |z|ˈ|iə|ɹ|oʊ| |t|ˈ|uː| |ˈ|eɪ|t| |ˈ|eɪ|t| "
        "|f|ˈ|oːɹ| |w|ˈ|ʌ|n| |n|ˈ|aɪ"
    ]


def test_grapheme_to_phoneme():
    """Test `grapheme_to_phoneme` against basic cases."""
    in_ = """of 5 stages:
(i) preparation,
(ii) incubation,
(iii) intimation,
(iv) illumination"""
    out = """ʌ_v_ _f_ˈ_aɪ_v_ _s_t_ˈ_eɪ_dʒ_ᵻ_z_:_
_(_ˈ_aɪ_)_ _p_ɹ_ˌ_ɛ_p_ɚ_ɹ_ˈ_eɪ_ʃ_ə_n_,_
_(_ɹ_ˌ_oʊ_m_ə_n_ _t_ˈ_uː_)_ _ˌ_ɪ_n_k_j_uː_b_ˈ_eɪ_ʃ_ə_n_,_
_(_ɹ_ˌ_oʊ_m_ə_n_ _θ_ɹ_ˈ_iː_)_ _ˌ_ɪ_n_t_ɪ_m_ˈ_eɪ_ʃ_ə_n_,_
_(_ɹ_ˌ_oʊ_m_ə_n_ _f_ˈ_oːɹ_)_ _ɪ_l_ˌ_uː_m_ᵻ_n_ˈ_eɪ_ʃ_ə_n"""
    assert grapheme_to_phoneme(in_, separator="_") == out
    out = "j_uː_ɹ_ˈ_iː_k_ɐ_ _w_ˈ_ɔː_k_s_ _ɑː_n_ð_ɪ_ _ˈ_ɛ_ɹ_ _ˈ_ɔː_l_ _ɹ_ˈ_aɪ_t_."
    assert grapheme_to_phoneme("Eureka walks on the air all right.", separator="_") == out
    assert grapheme_to_phoneme("Hello world", separator="_") == "h_ə_l_ˈ_oʊ_ _w_ˈ_ɜː_l_d"
    assert grapheme_to_phoneme("How are you?", separator="_") == "h_ˈ_aʊ_ _ɑːɹ_ _j_uː_?"
    assert grapheme_to_phoneme("I'm great!", separator="_") == "aɪ_m_ _ɡ_ɹ_ˈ_eɪ_t_!"
    assert grapheme_to_phoneme("") == ""


def test_grapheme_to_phoneme__empty():
    """Test `grapheme_to_phoneme` against an empty list."""
    assert grapheme_to_phoneme([]) == []


def test_grapheme_to_phoneme__doc_input():
    """Test `grapheme_to_phoneme` with a spaCy input."""
    nlp = lib.text.load_en_core_web_md(disable=("parser", "ner"))
    assert grapheme_to_phoneme(nlp("Hello world"), separator="_") == "h_ə_l_ˈ_oʊ_ _w_ˈ_ɜː_l_d"
    assert grapheme_to_phoneme([nlp("How are you?")], separator="_") == ["h_ˈ_aʊ_ _ɑːɹ_ _j_uː_?"]


def test_grapheme_to_phoneme__white_space():
    """Test `grapheme_to_phoneme` preserves white spaces, SOMETIMES."""
    out = " _ _h_ə_l_ˈ_oʊ_ _w_ˈ_ɜː_l_d_ _ "
    assert grapheme_to_phoneme("  Hello World  ", separator="_") == out

    out = " _\n_\n_ _h_ə_l_ˈ_oʊ_ _w_ˈ_ɜː_l_d_ _\n_\n_ "
    assert grapheme_to_phoneme(" \n\n Hello World \n\n ", separator="_") == out

    out = " _\n_\t_ _h_ə_l_ˈ_oʊ_ _w_ˈ_ɜː_l_d_ _\n_\t_ "
    assert grapheme_to_phoneme(" \n\t Hello World \n\t ", separator="_") == out

    # NOTE: Test a number of string literals, see: https://docs.python.org/2.0/ref/strings.html
    # TODO: Which of these punctuation marks need to be preserved?
    in_ = " \n test \t test \r test \v test \f test \a test \b "
    out = " _\n_ _t_ˈ_ɛ_s_t_ _t_ˈ_ɛ_s_t_ _t_ˈ_ɛ_s_t_ _t_ˈ_ɛ_s_t_ _t_ˈ_ɛ_s_t_ _t_ˈ_ɛ_s_t_ "
    assert grapheme_to_phoneme(in_, separator="_") == out

    # TODO: Multiple white-spaces are not preserved.
    in_ = "résumé résumé  résumé   résumé"
    out = "ɹ|ˈ|ɛ|z|uː|m|ˌ|eɪ| |ɹ|ˈ|ɛ|z|uː|m|ˌ|eɪ| |ɹ|ˈ|ɛ|z|uː|m|ˌ|eɪ| |ɹ|ˈ|ɛ|z|uː|m|ˌ|eɪ"
    assert grapheme_to_phoneme(in_, separator="|") == out


def test_grapheme_to_phoneme__unique_separator():
    """Test `grapheme_to_phoneme` errors if `separator` is not unique."""
    with pytest.raises(AssertionError):
        grapheme_to_phoneme("Hello World!", separator="!")


def test_grapheme_to_phoneme__spacy_failure_cases():
    """Test `grapheme_to_phoneme` with text where spaCy fails."""
    in_ = [
        ".Sameer M Babu is a professor who wrote an article about classroom "
        "climate and social intelligence.",
        "I ha thought till my brains ached,-Beli me, John, I have. An I say again, theres no "
        "help for us but having faith i the Union. Theyll win the day, see if they dunnot!",
        "we're gett'n' a long way from home. And see how the clouds are rolling just above us, "
        "remarked the boy, who was almost as uneasy as captain Bill.",
        "I I don't s-s-see any-thing funny 'bout it! he stammered.",
        "But don't worry, there are plenty of toys that are safe--and fun--for your child.",
        "Indeed, for the royal body, a rather unusual set of ideal attributes emerges in the "
        "Mesopotamian lexicon: an accumulation of good form or breeding, auspiciousness, "
        "vigor/vitality, and, specifically, sexual allure or charm – all of which are not "
        "only ascribed in text, but equally to be read in imagery.",
        "This is a test {}:\"?,./;'[]\\q234567890-=!@##$%%^&*()+",
    ]
    out = [
        "d_ˈ_ɑː_t_ _s_æ_m_ˈ_ɪ_ɹ_ _ˈ_ɛ_m_ _b_ˈ_ɑː_b_uː_ _ɪ_z_ _ɐ_ _p_ɹ_ə_f_ˈ_ɛ_s_ɚ_ _h_ˌ_uː_ "
        "_ɹ_ˈ_oʊ_t_ _ɐ_n_ _ˈ_ɑːɹ_ɾ_ɪ_k_əl_ _ɐ_b_ˌ_aʊ_t_ _k_l_ˈ_æ_s_ɹ_uː_m_ _k_l_ˈ_aɪ_m_ə_t_ "
        "_æ_n_d_ _s_ˈ_oʊ_ʃ_əl_ _ɪ_n_t_ˈ_ɛ_l_ɪ_dʒ_ə_n_s_.",
        "aɪ_ _h_ˈ_ɑː_ _θ_ˈ_ɔː_t_ _t_ˈ_ɪ_l_ _m_aɪ_ _b_ɹ_ˈ_eɪ_n_z_ _ˈ_eɪ_k_t_b_ɪ_l_i_ _m_ˌ_iː_,_ _"
        "dʒ_ˈ_ɑː_n_,_ _aɪ_ _h_ˈ_æ_v_._ _ɐ_n_ _aɪ_ _s_ˈ_eɪ_ _ɐ_ɡ_ˈ_ɛ_n_,_ _ð_ɚ_z_ _n_ˈ_oʊ_ "
        "_h_ˈ_ɛ_l_p_ _f_ɔː_ɹ_ _ˌ_ʌ_s_ _b_ˌ_ʌ_t_ _h_ˌ_æ_v_ɪ_ŋ_ _f_ˈ_eɪ_θ_ _ˈ_aɪ_ "
        "_ð_ə_ _j_ˈ_uː_n_iə_n_._ _θ_ˈ_eɪ_l_ _w_ˈ_ɪ_n_ _ð_ə_ _d_ˈ_eɪ_,_ _s_ˈ_iː_ _ɪ_f_ _ð_eɪ_ "
        "_d_ˈ_ʌ_n_ɑː_t_!",
        "w_ɪɹ_ _ɡ_ˈ_ɛ_t_n_ _ɐ_ _l_ˈ_ɑː_ŋ_ _w_ˈ_eɪ_ _f_ɹ_ʌ_m_ _h_ˈ_oʊ_m_._ _æ_n_d_ _s_ˈ_iː_ _"
        "h_ˌ_aʊ_ _ð_ə_ _k_l_ˈ_aʊ_d_z_ _ɑːɹ_ _ɹ_ˈ_oʊ_l_ɪ_ŋ_ _dʒ_ˈ_ʌ_s_t_ _ə_b_ˈ_ʌ_v_ _ˌ_ʌ_s_,_ "
        "_ɹ_ɪ_m_ˈ_ɑːɹ_k_t_ _ð_ə_ _b_ˈ_ɔɪ_,_ _h_ˌ_uː_ _w_ʌ_z_ _ˈ_ɔː_l_m_oʊ_s_t_ _"
        "æ_z_ _ʌ_n_ˈ_iː_z_i_ _æ_z_ _k_ˈ_æ_p_t_ɪ_n_ _b_ˈ_ɪ_l_.",
        "aɪ_ _aɪ_ _d_ˈ_oʊ_n_t_ _ˈ_ɛ_s_-_ˈ_ɛ_s_-_s_ˈ_iː_ _ˌ_ɛ_n_i_-_θ_ˈ_ɪ_ŋ_ _"
        "f_ˈ_ʌ_n_i_ _b_ˈ_aʊ_t_ _ɪ_t_!_ _h_iː_ _s_t_ˈ_æ_m_ɚ_d_.",
        "b_ˌ_ʌ_t_ _d_ˈ_oʊ_n_t_ _w_ˈ_ʌ_ɹ_i_,_ _ð_ɛ_ɹ_ˌ_ɑːɹ_ _p_l_ˈ_ɛ_n_t_i_ _ʌ_v_ _"
        "t_ˈ_ɔɪ_z_ _ð_æ_t_ _ɑːɹ_ _s_ˈ_eɪ_f_-_-_æ_n_d_ _f_ˈ_ʌ_n_-_-_f_ɔːɹ_ _j_ʊɹ_ _tʃ_ˈ_aɪ_l_d_.",
        "ˌ_ɪ_n_d_ˈ_iː_d_,_ _f_ɚ_ð_ə_ _ɹ_ˈ_ɔɪ_əl_ _b_ˈ_ɑː_d_i_,_ _ɐ_ _ɹ_ˈ_æ_ð_ɚ_ɹ_ "
        "_ʌ_n_j_ˈ_uː_ʒ_uː_əl_ _s_ˈ_ɛ_t_ _ʌ_v_ _aɪ_d_ˈ_iə_l_ _ˈ_æ_t_ɹ_ɪ_b_j_ˌ_uː_t_s_ "
        "_ɪ_m_ˈ_ɜː_dʒ_ᵻ_z_ _ɪ_n_ð_ə_ _m_ˌ_ɛ_s_ə_p_ə_t_ˈ_eɪ_m_iə_n_ _l_ˈ_ɛ_k_s_ɪ_k_ə_n_:_ "
        "_ɐ_n_ _ɐ_k_j_ˌ_uː_m_j_ʊ_l_ˈ_eɪ_ʃ_ə_n_ _ʌ_v_ _ɡ_ˈ_ʊ_d_ _f_ˈ_ɔːɹ_m_ _ɔːɹ_ "
        "_b_ɹ_ˈ_iː_d_ɪ_ŋ_,_ "
        "_ɔː_s_p_ˈ_ɪ_ʃ_ə_s_n_ə_s_,_ _v_ˈ_ɪ_ɡ_ɚ_ _s_l_ˈ_æ_ʃ_ _v_aɪ_t_ˈ_æ_l_ɪ_ɾ_i_,_ _æ_n_d_,_ "
        "_s_p_ə_s_ˈ_ɪ_f_ɪ_k_l_i_,_ _s_ˈ_ɛ_k_ʃ_uː_əl_ _ɐ_l_ˈ_ʊɹ_ _ɔːɹ_ _tʃ_ˈ_ɑːɹ_m_ "
        "_–_ _ˈ_ɔː_l_ _ʌ_v_w_ˈ_ɪ_tʃ_ _ɑːɹ_ _n_ˌ_ɑː_t_ _ˈ_oʊ_n_l_i_ _ɐ_s_k_ɹ_ˈ_aɪ_b_d_ _ɪ_n_ "
        "_t_ˈ_ɛ_k_s_t_,_ _b_ˌ_ʌ_t_ _ˈ_iː_k_w_əl_i_ _t_ə_b_i_ _ɹ_ˈ_ɛ_d_ _ɪ_n_ _ˈ_ɪ_m_ɪ_dʒ_ɹ_i_.",
        'ð_ɪ_s_ _ɪ_z_ _ɐ_ _t_ˈ_ɛ_s_t_ _{_}_:_"_?_,_d_ˈ_ɑː_t_s_l_æ_ʃ_ _b_ˈ_æ_k_s_l_æ_ʃ_ _k_j_ˈ_uː_ '
        "_t_ˈ_uː_h_ˈ_ʌ_n_d_ɹ_ə_d_ _θ_ˈ_ɜː_ɾ_i_f_ˈ_oːɹ_ _m_ˈ_ɪ_l_iə_n_ _f_ˈ_aɪ_v_h_ˈ_ʌ_n_d_ɹ_ə_d_"
        " _s_ˈ_ɪ_k_s_t_i_s_ˈ_ɛ_v_ə_n_ _θ_ˈ_aʊ_z_ə_n_d_ _ˈ_eɪ_t_h_ˈ_ʌ_n_d_ɹ_ə_d_ _n_ˈ_aɪ_n_t_i_"
        " _ˌ_iː_k_w_əl_z_ˌ_ɛ_k_s_k_l_ə_m_ˌ_eɪ_ʃ_ə_n_ˌ_æ_t_h_ɐ_ʃ_h_ˌ_æ_ʃ_d_ə_l_ɚ_p_ɚ_s_ˈ_ɛ_n_t_p"
        "_ɚ_s_ˈ_ɛ_n_t_ɐ_n_d_ˌ_æ_s_t_ɚ_ɹ_ˌ_ɪ_s_k_p_l_ʌ_s",
    ]
    assert grapheme_to_phoneme(in_, separator="_") == out


def test__load_amepd():
    """Test `lib.text._load_amepd` loads the dictionary, and the dictionary includes all the
    part-of-speech, phonemes, fields, etc."""
    dictionary = lib.text._load_amepd()
    part_of_speech_coarse = set()
    part_of_speech_fine = set()
    arpabet: typing.Set[lib.text.AMEPD_ARPABET] = set()
    fields = set()
    unique = set()
    characters = set()
    for word, pronunciations in dictionary.items():
        for pronunciation in pronunciations:
            assert len(word) == len(word.strip())
            assert len(word) > 0
            if pronunciation.pos is not None:
                part_of_speech_coarse.add(pronunciation.pos.coarse)
                part_of_speech_fine.add(pronunciation.pos.fine)
            arpabet.update(pronunciation.pronunciation)
            characters.update(list(word))
            signature = (
                word,
                pronunciation.pronunciation,
                pronunciation.pos,
                pronunciation.metadata.usage,
            )
            assert signature not in unique
            unique.add(signature)
            for field in lib.text.AmEPDMetadata._fields:
                value = getattr(pronunciation.metadata, field)
                if value is not None:
                    fields.add(field)
                    for subvalue in value if isinstance(value, tuple) else [value]:
                        assert len(subvalue) == len(subvalue.strip())
                        assert len(subvalue) > 0
    assert len(fields) == len(lib.text.AmEPDMetadata._fields)
    assert part_of_speech_coarse == set(get_args(lib.text.AMEPD_PART_OF_SPEECH_COARSE))
    assert part_of_speech_fine == set(list(get_args(lib.text.AMEPD_PART_OF_SPEECH_FINE)) + [None])
    assert arpabet == set(get_args(lib.text.AMEPD_ARPABET))
    assert characters == set(list(string.ascii_uppercase) + ["'"])


def _check_pronunciation(
    word: str,
    part_of_speech_coarse: typing.Optional[
        typing.Literal[lib.text.AMEPD_PART_OF_SPEECH_COARSE]
    ] = None,
    part_of_speech_fine: typing.Optional[typing.Literal["past", "pres"]] = None,
    expected: typing.Optional[str] = None,
):
    result = lib.text.get_pronunciation(word, part_of_speech_coarse, part_of_speech_fine)
    if expected is None:
        assert result is expected
    else:
        assert result == tuple(expected.split())


def test_get_pronunciation__out_of_vocabulary():
    """Test `lib.text.get_pronunciation` doesn't handle words outside it's vocabulary."""
    _check_pronunciation("abcdefg", expected=None)


def test_get_pronunciation__apostrophes():
    """Test `lib.text.get_pronunciation` handles apostrophes at the end and beginning of a word."""
    _check_pronunciation("accountants'", "verb", "past", expected="AX K AW1 N T AX N T S")
    _check_pronunciation("'bout", expected="B AW1 T")


def test_get_pronunciation__disambiguate():
    """Test `lib.text.get_pronunciation` attempts to disambiguate tricky cases, and returns `None`
    when it can't."""
    # NOTE: Base case with no variations to choose from.
    _check_pronunciation("fly", "verb", "pres", expected="F L AY1")
    _check_pronunciation("fly", "verb", "past", expected="F L AY1")
    _check_pronunciation("fly", "verb", expected="F L AY1")
    _check_pronunciation("fly", "noun", expected="F L AY1")
    _check_pronunciation("fly", expected="F L AY1")
    _check_pronunciation("fly", expected="F L AY1")

    _check_pronunciation("read", expected=None)  # Options: verb@past, verb
    _check_pronunciation("read", "verb", expected=None)
    _check_pronunciation("beloved", expected=None)  # Options: noun, adj@attr, adj@pred, verb
    _check_pronunciation("beloved", "adj", expected=None)
    # NOTE: Multiple variations that cannot be disambiguated with part-of-speech.
    _check_pronunciation("reasonable", expected=None)  # Options: 1, 2
    _check_pronunciation("reasonable", "adj", expected=None)

    # NOTE: This should be disambiguated correctly.
    _check_pronunciation("read", "verb", "past", expected="R EH1 D")
    _check_pronunciation("read", "verb", "pres", expected="R IY1 D")
    _check_pronunciation("beloved", "verb", expected="B IH0 L AH1 V D")
    _check_pronunciation("beloved", "noun", expected="B IH0 L AH1 V D")

    # NOTE: Abbreviations that are sometimes expanded during voice-over are not disambiguated.
    _check_pronunciation("feb", "noun", expected=None)


def test_get_initialism_pronunciation():
    """Test `lib.text.get_initialism_pronunciation` handles initialisms with various letters and
    cases."""
    assert lib.text.get_initialism_pronunciation("ibm") == tuple("AY1 B IY1 EH1 M".split())
    assert lib.text.get_initialism_pronunciation("IBM") == tuple("AY1 B IY1 EH1 M".split())
    assert lib.text.get_initialism_pronunciation("CACLD") == tuple(
        "S IY1 EY1 S IY1 EH1 L D IY1".split()
    )
    assert lib.text.get_initialism_pronunciation("FOIA") == tuple("EH1 F OW1 AY1 EY1".split())
    assert lib.text.get_initialism_pronunciation("GENEGO") == tuple(
        "JH IY1 IY1 EH1 N IY1 JH IY1 OW1".split()
    )
    assert lib.text.get_initialism_pronunciation("SRI") == tuple("EH1 S AA1 R AY1".split())
    assert lib.text.get_initialism_pronunciation("USA") == tuple("Y UW1 EH1 S EY1".split())


def test_get_pronunciation__non_standard_words():
    """Test `lib.text.get_pronunciation` errors given non-standard words."""
    with pytest.raises(AssertionError):
        lib.text.get_pronunciation("I B M")
    with pytest.raises(AssertionError):
        lib.text.get_pronunciation("I.B.M.")
    with pytest.raises(AssertionError):
        lib.text.get_pronunciation("able-bodied")
    with pytest.raises(AssertionError):
        lib.text.get_pronunciation("ABC123")


def test_get_pronunciations():
    """Test `lib.text.get_pronunciations` against basic cases: non-standard words, initialisms,
    appostrophes, and abbreviations."""
    nlp = lib.text.load_en_core_web_md()
    get_pronunciations = lambda s: lib.text.get_pronunciations(nlp(s))
    assert get_pronunciations("In 1968 the U.S. Army") == (
        ("IH1", "N"),
        None,  # Non-standard word ignored
        None,
        None,  # Non-standard word ignored
        ("AA1", "R", "M", "IY0"),
    )
    assert get_pronunciations("Individual-Based Model (IBM)") == (
        ("IH2", "N", "D", "IH0", "V", "IH1", "JH", "UW0", "AX", "L"),
        None,
        ("B", "EY1", "S", "T"),
        ("M", "AA1", "D", "AX", "L"),
        None,
        ("AY1", "B", "IY1", "EH1", "M"),  # Initialism handled
        None,
    )
    assert get_pronunciations("NASA's TV mission is to pioneer.") == (
        ("N", "AE1", "S", "AX"),
        None,  # NOTE: spaCy splits up apostrophes
        None,  # Ambigious abbreviation is ignored
        ("M", "IH1", "SH", "AX", "N"),
        ("IH1", "Z"),
        None,
        ("P", "AY2", "AX", "N", "IH1", "R"),
        None,
    )
    assert get_pronunciations("Youssou N'Dour is a Senegalese singer") == (
        None,
        ("N", "D", "AW1", "AXR"),  # Apostrophes handled
        ("IH1", "Z"),
        None,
        ("S", "EH2", "N", "AX", "G", "AX", "L", "IY1", "Z"),
        ("S", "IH1", "NG", "G", "AXR"),
    )


def test_get_pronunciations__part_of_speech():
    """Test `lib.text.get_pronunciations` with ambigious part of speech cases."""
    nlp = lib.text.load_en_core_web_md()
    get_pronunciations = lambda s: lib.text.get_pronunciations(nlp(s))
    assert get_pronunciations("It was time to present the present.") == (
        ("IH1", "T"),
        None,
        ("T", "AY1", "M"),
        None,
        ("P", "R", "IH0", "Z", "EH1", "N", "T"),  # Verb
        None,
        ("P", "R", "EH1", "Z", "AX", "N", "T"),  # Noun
        None,
    )
    assert get_pronunciations("He has read the whole thing.") == (
        ("HH", "IY1"),
        None,
        ("R", "EH1", "D"),  # Verb, Past Tense
        None,
        ("HH", "OW1", "L"),
        ("TH", "IH1", "NG"),
        None,
    )
    assert get_pronunciations("We read.") == (
        ("W", "IY1"),
        ("R", "IY1", "D"),  # Verb, Present Tense
        None,
    )


def test_natural_keys():
    """Test `lib.text.natural_keys` sorts naturally."""
    list_ = ["name 0", "name 1", "name 10", "name 11"]
    assert sorted(list_, key=lib.text.natural_keys) == list_


def test_strip():
    """Test `lib.text.strip` handles various white space scenarios."""
    assert lib.text.strip("  Hello World  ") == ("Hello World", "  ", "  ")
    assert lib.text.strip("Hello World  ") == ("Hello World", "", "  ")
    assert lib.text.strip("  Hello World") == ("Hello World", "  ", "")
    assert lib.text.strip(" \n Hello World \n ") == ("Hello World", " \n ", " \n ")
    assert lib.text.strip(" \n\n Hello World \n\n ") == (
        "Hello World",
        " \n\n ",
        " \n\n ",
    )


_GERMAN_SPECIAL_CHARACTERS = ["ß", "ä", "ö", "ü", "Ä", "Ö", "Ü", "«", "»", "—"]


def test_normalize_vo_script():
    """Test `lib.text.normalize_vo_script` handles all characters from 0 - 128. And
    test `lib.text.normalize_vo_script` handles all German special characters when decode=False."""
    # fmt: off
    assert list(lib.text.normalize_vo_script(chr(i), strip=False) for i in range(0, 128)) == [
        "", "", "", "", "", "", "", "", "", "  ", "\n", "", "\n", "\n", "", "", "", "", "", "", "",
        "", "", "", "", "", "", "", "", "", "", "", " ", "!", '"', "#", "$", "%", "&", "'", "(",
        ")", "*", "+", ",", "-", ".", "/", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", ":",
        ";", "<", "=", ">", "?", "@", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L",
        "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "[", "\\", "]", "^",
        "_", "`", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p",
        "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "{", "|", "}", "~", ""
    ]
    assert list(lib.text.normalize_vo_script(c, strip=False, decode=False) for c in _GERMAN_SPECIAL_CHARACTERS) == _GERMAN_SPECIAL_CHARACTERS
    # fmt: on


def test_is_normalized_vo_script():
    """Test `lib.text.is_normalized_vo_script` handles all characters from 0 - 128.
    And test `lib.text.is_normalized_vo_script` handles all German special characters."""
    assert all(
        lib.text.is_normalized_vo_script(lib.text.normalize_vo_script(chr(i), strip=False))
        for i in range(0, 128)
    )
    assert all(
        lib.text.is_normalized_vo_script(lib.text.normalize_vo_script(c, strip=False))
        for c in _GERMAN_SPECIAL_CHARACTERS
    )


def test_is_normalized_vo_script__unnormalized():
    """Test `lib.text.is_normalized_vo_script` fails for unnormalized characters."""
    # fmt: off
    assert [(chr(i), lib.text.is_normalized_vo_script(chr(i))) for i in range(0, 128)] == [
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
    """Test `lib.text.is_voiced` handles all characters and an empty string."""
    assert lib.text.is_voiced("123")
    assert lib.text.is_voiced("abc")
    assert lib.text.is_voiced("ABC")
    for char in "@#$%&+=*".split():
        assert lib.text.is_voiced(char)
    assert not lib.text.is_voiced("!^()_{[}]:;\"'<>?/~`|\\")
    assert not lib.text.is_voiced("")


def test_has_digit():
    """Test `lib.text.has_digit` handles basic cases."""
    assert lib.text.has_digit("123")
    assert lib.text.has_digit("123abc")
    assert lib.text.has_digit("1")
    assert not lib.text.has_digit("abc")
    assert not lib.text.has_digit("")


def test_add_space_between_sentences():
    """Test `lib.text.add_space_between_sentences` adds a space between sentences."""
    nlp = lib.text.load_en_core_web_md(disable=("tagger", "ner"))
    script = (
        "Business was involved in slavery, colonialism, and the cold war.The term "
        "'business ethics' came into common use in the United States in the early 1970s."
    )
    assert lib.text.add_space_between_sentences(nlp(script)) == (
        "Business was involved in slavery, colonialism, and the cold war. The term "
        "'business ethics' came into common use in the United States in the early 1970s."
    )
    script = (
        "Mix and match the textured shades for a funky effect.Hang on "
        "to these fuzzy hangers from Domis."
    )
    assert lib.text.add_space_between_sentences(nlp(script)) == (
        "Mix and match the textured shades for a funky effect. Hang on "
        "to these fuzzy hangers from Domis."
    )


def test_add_space_between_sentences__new_lines():
    """Test `lib.text.add_space_between_sentences` adds a space between sentences while handling
    newlines."""
    nlp = lib.text.load_en_core_web_md(disable=("tagger", "ner"))
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
    assert lib.text.add_space_between_sentences(nlp(script)) == expected


def test_add_space_between_sentences__one_word():
    """Test `lib.text.add_space_between_sentences` handles one word."""
    nlp = lib.text.load_en_core_web_md(disable=("tagger", "ner"))
    assert lib.text.add_space_between_sentences(nlp("Hi")) == "Hi"
    assert lib.text.add_space_between_sentences(nlp("Hi  ")) == "Hi  "
    assert lib.text.add_space_between_sentences(nlp("Hi.  ")) == "Hi.  "


def test_add_space_between_sentences__regression():
    """Test `lib.text.add_space_between_sentences` handles these regression tests on Hilary's
    data."""
    nlp = lib.text.load_en_core_web_md(disable=("tagger", "ner"))
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
        assert lib.text.add_space_between_sentences(nlp(script)) == script


def test_normalize_non_standard_words():
    cases = [
        ("Mr. Gurney", "Mr. Gurney"),
        ("San Antonio at 1:30 p.m.,", "San Antonio at one thirty P M,"),
        ("between May 1st, 1827,", "between May first, eighteen twenty seven,"),
        (
            "inch BBL, unquote, cost $29.95.",
            "inch B B L, unquote, cost twenty nine point nine five dollars.",
        ),
        (
            "Post Office Box 2915, Dallas, Texas",
            "Post Office B O X two thousand, nine hundred and fifteen, Dallas, Texas",
        ),
        (
            "serial No. C2766, which was also found",
            "serial No. century two thousand, seven hundred and sixty six, which was also found",
        ),
        ("Newgate down to 1818,", "Newgate down to eighteen eighteen,"),
        (
            "It was about 250 B.C., when the great",
            "It was about two hundred and fifty B C, when the great",
        ),
        ("In 606, Nineveh", "In six hundred and six, Nineveh"),
        ("Exhibit No. 143 as the", "Exhibit No. one hundred and forty three as the"),
        ("William IV. was also the victim", "William the fourth. was also the victim"),
        ("Chapter 4. The Assassin:", "Chapter four. The Assassin:"),
        ("was shipped on March 20, and the", "was shipped on March twentieth, and the"),
        ("4 March 2014", "the fourth of March twenty fourteen"),
        (
            "distance of 265.3 feet was, quote",
            "distance of two hundred and sixty five point three feet was, quote",
        ),
        (
            "information on some 50,000 cases",
            "information on some fifty thousand cases",
        ),
        (
            "PRS received items in 8,709 cases",
            "P R S received items in eight thousand, seven hundred and nine cases",
        ),
    ]
    for input_, output in cases:
        assert lib.text.normalize_non_standard_words(input_) == output


def _align_and_format(tokens, other, **kwargs):
    cost, alignment = lib.text.align_tokens(tokens, other, **kwargs)
    return lib.text.format_alignment(tokens, other, alignment)


def test_align_tokens__empty():
    """Test `lib.text.align_tokens` aligns empty text correctly."""
    assert lib.text.align_tokens("", "")[0] == 0
    assert lib.text.align_tokens("a", "")[0] == 1
    assert lib.text.align_tokens("", "a")[0] == 1
    assert lib.text.align_tokens("abc", "")[0] == 3
    assert lib.text.align_tokens("", "abc")[0] == 3
    assert lib.text.align_tokens("", "abc", window_length=1)[0] == 3


def test_align_tokens__one_letter():
    """Test `lib.text.align_tokens` aligns one letter correctly."""
    # Should just add "a" to the beginning.
    assert lib.text.align_tokens("abc", "bc", window_length=1)[0] == 1
    assert lib.text.align_tokens("abc", "bc", allow_substitution=lambda a, b: False)[0] == 1
    assert _align_and_format("abc", "bc") == (
        "a b c",
        "  b c",
    )

    # Should just add I to the beginning.
    assert lib.text.align_tokens("islander", "slander")[0] == 1
    assert lib.text.align_tokens("islander", "slander", window_length=1)[0] == 1
    assert _align_and_format("islander", "slander") == (
        "i s l a n d e r",
        "  s l a n d e r",
    )


def test_align_tokens__deletion():
    """Test `lib.text.align_tokens` deletion."""
    # Should delete 4 letters FOOT at the beginning.
    assert lib.text.align_tokens("football", "foot")[0] == 4


def test_align_tokens__substitution():
    """Test `lib.text.align_tokens` substitution."""
    # Needs to substitute the first 5 chars: INTEN by EXECU
    assert lib.text.align_tokens("intention", "execution")[0] == 5


def test_align_tokens__multi_operation_alignments():
    """Test `lib.text.align_tokens` substitution, insertion, and deletion."""
    # Needs to substitute M by K, T by M and add an A to the end
    assert lib.text.align_tokens("mart", "karma")[0] == 3

    # Needs to substitute K by M, M by T, and delete A from the end
    assert lib.text.align_tokens("karma", "mart")[0] == 3

    # Substitute K by S, E by I and add a G at the end.
    assert lib.text.align_tokens("kitten", "sitting")[0] == 3


def test_align_tokens__window_lengths():
    """Test `lib.text.align_tokens` handles various window lengths."""
    assert lib.text.align_tokens("ball", "football")[0] == 4
    assert lib.text.align_tokens("ball", "football", window_length=1)[0] == 7
    assert _align_and_format("ball", "football", window_length=1) == (
        "b a l       l",
        "f o o t b a l",
    )
    assert lib.text.align_tokens("ball", "football", window_length=2)[0] == 6
    assert _align_and_format("ball", "football", window_length=2) == (
        "b a         l l",
        "f o o t b a l l",
    )
    assert lib.text.align_tokens("ball", "football", window_length=3)[0] == 4
    assert _align_and_format("ball", "football", window_length=3) == (
        "        b a l l",
        "f o o t b a l l",
    )


def test_align_tokens__word_subtitution():
    """Test `lib.text.align_tokens` substitutes words."""
    assert lib.text.align_tokens(["Hey", "There"], ["Hey", "There"])[0] == 0
    assert lib.text.align_tokens(["Hey", "There"], ["Hi", "There"])[0] == 2
    assert lib.text.align_tokens(["Hey", "There"], ["Hi", "The"])[0] == 4


def test_align_tokens__word_deletion():
    """Test `lib.text.align_tokens` deletes words."""
    assert lib.text.align_tokens(["Hey", "There", "You"], ["Hey", ",", "There"])[0] == 4
    assert _align_and_format(["Hey", "There", "You"], ["Hey", ",", "There"]) == (
        "Hey   There",
        "Hey , There",
    )

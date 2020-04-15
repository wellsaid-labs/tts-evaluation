import pytest

from src.datasets import HILARY_NORIEGA
from src.datasets import JUDY_BIEBER
from src.datasets import MARY_ANN
from src.spectrogram_model import InputEncoder
from src.spectrogram_model.input_encoder import _grapheme_to_phoneme
from src.spectrogram_model.input_encoder import _grapheme_to_phoneme_perserve_punctuation
from src.spectrogram_model.input_encoder import cache_grapheme_to_phoneme_perserve_punctuation
from src.utils.disk_cache_ import make_arg_key


def test_cache_grapheme_to_phoneme_perserve_punctuation():
    cache_grapheme_to_phoneme_perserve_punctuation(['Hello world'], separator='_')
    cache_grapheme_to_phoneme_perserve_punctuation(['How are you?'], separator='_')
    cache_grapheme_to_phoneme_perserve_punctuation(['I\'m great!'], separator='_')

    get_result = lambda s: _grapheme_to_phoneme_perserve_punctuation.disk_cache.get(
        make_arg_key(_grapheme_to_phoneme_perserve_punctuation.__wrapped__, s, separator='_'))

    assert get_result('Hello world') == 'h_ə_l_ˈ_oʊ_ _w_ˈ_ɜː_l_d'
    assert get_result('How are you?') == 'h_ˈ_aʊ_ _ɑːɹ_ _j_uː_?'
    assert get_result('I\'m great!') == 'aɪ_m_ _ɡ_ɹ_ˈ_eɪ_t_!'


def test__grapheme_to_phoneme__separator():
    # Test using `service_separator`.
    assert _grapheme_to_phoneme('Hello World', separator='_') == 'h_ə_l_ˈ_oʊ_ _w_ˈ_ɜː_l_d'

    with pytest.raises(AssertionError):  # Test separator is not unique
        _grapheme_to_phoneme('Hello World', separator='ə')


def test__grapheme_to_phoneme_perserve_punctuation__separator():
    with pytest.raises(AssertionError):  # Test separator is not unique
        _grapheme_to_phoneme_perserve_punctuation('Hello World!', separator='!')


def test__grapheme_to_phoneme():
    grapheme = [
        "  Hello World  ",  # Test stripping
        "Hello World  ",  # Test stripping
        "  Hello World",  # Test stripping
        " \n Hello World \n ",  # Test stripping
        " \n\n Hello World \n\n ",  # Test stripping
        '"It is commonly argued that the notion of',  # Test bash escape
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
    phoneme = [
        ' _ _h_ə_l_ˈ_oʊ_ _w_ˈ_ɜː_l_d_ _ ',
        'h_ə_l_ˈ_oʊ_ _w_ˈ_ɜː_l_d_ _ ',
        ' _ _h_ə_l_ˈ_oʊ_ _w_ˈ_ɜː_l_d',
        ' _\n_ _h_ə_l_ˈ_oʊ_ _w_ˈ_ɜː_l_d_ _\n_ ',
        ' _\n_\n_ _h_ə_l_ˈ_oʊ_ _w_ˈ_ɜː_l_d_ _\n_\n_ ',
        'ɪ_t_ _ɪ_z_ _k_ˈ_ɑː_m_ə_n_l_i_ _ˈ_ɑːɹ_ɡ_j_uː_d_ _ð_æ_t_ð_ə_ _n_ˈ_oʊ_ʃ_ə_n_ _ʌ_v',
        'æ_n_d_ _t_ɹ_ˈ_ɑː_t_ _f_ˈ_ɛ_d_ _ɪ_t_ _ɐ_ _h_ˈ_æ_n_d_f_əl_ _ʌ_v_ _f_ɹ_ˈ_ɛ_ʃ_ _b_l_ˈ_uː_ _'
        'k_l_ˈ_oʊ_v_ɚ_ _æ_n_d_ _s_m_ˈ_uː_ð_d_ _æ_n_d_ _p_ˈ_ɛ_ɾ_ᵻ_d_ _ɪ_t_ _ʌ_n_t_ˈ_ɪ_l_ _ð_ə_ _'
        'l_ˈ_æ_m_ _w_ʌ_z_ _ˈ_iː_ɡ_ɚ_ _t_ə_ _f_ˈ_ɑː_l_oʊ_ _h_ɜː_ _w_ɛɹ_ɹ_ˈ_ɛ_v_ɚ_ _ʃ_iː_ _m_ˌ_aɪ_t_ '
        '_ɡ_ˈ_oʊ',
        'ð_ə_ _h_ˈ_æ_b_ɪ_t_s_ _ʌ_v_ _m_ˈ_aɪ_n_d_ _ð_æ_t_ _k_ˈ_æ_ɹ_ɪ_k_t_ɚ_ɹ_ˌ_aɪ_z_ _ɐ_ _'
        'p_ˈ_ɜː_s_ə_n_ _s_t_ɹ_ˈ_ɔ_ŋ_l_i_ _d_ɪ_s_p_ˈ_oʊ_z_d_ _t_ʊ_w_ˈ_ɔːɹ_d_ _'
        'k_ɹ_ˈ_ɪ_ɾ_ɪ_k_əl_ _θ_ˈ_ɪ_ŋ_k_ɪ_ŋ_ _ɪ_n_k_l_ˈ_uː_d_ _ɐ_ _d_ɪ_z_ˈ_aɪɚ_ _t_ə_ _'
        'f_ˈ_ɑː_l_oʊ_ _ɹ_ˈ_iː_z_ə_n_ _æ_n_d_ _ˈ_ɛ_v_ɪ_d_ə_n_s_ _w_ɛɹ_ɹ_ˈ_ɛ_v_ɚ_ _ð_eɪ_ _'
        'm_ˈ_eɪ_ _l_ˈ_iː_d',
        'ð_ə_ _h_ˈ_æ_b_ɪ_t_s_ _ʌ_v_ _m_ˈ_aɪ_n_d_ _ð_æ_t_ _k_ˈ_æ_ɹ_ɪ_k_t_ɚ_ɹ_ˌ_aɪ_z_ _ɐ_ _'
        'p_ˈ_ɜː_s_ə_n_ _s_t_ɹ_ˈ_ɔ_ŋ_l_i_ _d_ɪ_s_p_ˈ_oʊ_z_d_ _t_ʊ_w_ˈ_ɔːɹ_d_ _'
        'k_ɹ_ˈ_ɪ_ɾ_ɪ_k_əl_ _θ_ˈ_ɪ_ŋ_k_ɪ_ŋ_ _ɪ_n_k_l_ˈ_uː_d_ _ɐ_ _d_ɪ_z_ˈ_aɪɚ_ _t_ə_ _'
        'f_ˈ_ɑː_l_oʊ_ _ɹ_ˈ_iː_z_ə_n_ _æ_n_d_ _ˈ_ɛ_v_ɪ_d_ə_n_s_ _w_ɛɹ_ɹ_ˈ_ɛ_v_ɚ_ _'
        'ð_eɪ_ _m_ˈ_eɪ_ _l_ˈ_iː_d_ _ɐ_ _s_ˌ_ɪ_s_t_ə_m_ˈ_æ_ɾ_ɪ_k',
        'b_ˌ_ʌ_t_ _w_ɛɹ_ɹ_ˈ_ɛ_v_ɚ_ _ð_eɪ_ _f_ˈ_ɔː_t_ _ɪ_n_ _n_ˈ_ɔːɹ_θ_ _ˈ_æ_f_ɹ_ɪ_k_ə_ _ɔːɹ_ _'
        'ð_ə_ _s_ˈ_aʊ_θ_ _p_ɐ_s_ˈ_ɪ_f_ɪ_k_ _ɔːɹ_ _w_ˈ_ɛ_s_t_ɚ_n_ _j_ˈ_ʊ_ɹ_ə_p_ _ð_ɪ_ _'
        'ˈ_ɪ_n_f_ə_n_t_ɹ_i_ _b_ˈ_oːɹ_ _ð_ə_ _b_ɹ_ˈ_ʌ_n_t_ _ʌ_v_ð_ə_ _f_ˈ_aɪ_ɾ_ɪ_ŋ_ _'
        'ɑː_n_ð_ə_ _ɡ_ɹ_ˈ_aʊ_n_d_ _æ_n_d_ _s_ˈ_ɛ_v_ə_n_ _ˌ_aʊ_ɾ_ə_v_ _t_ˈ_ɛ_n_ _s_ˈ_ʌ_f_ɚ_d_ _'
        'k_ˈ_æ_ʒ_uː_əl_ɾ_ɪ_z',
        'aɪ_ _l_ˈ_eɪ_ _ˈ_ɛ_ɡ_z_ _w_ɛɹ_ɹ_ˈ_ɛ_v_ɚ_ɹ_ _aɪ_ _h_ˈ_æ_p_ə_n_ _t_ə_ _b_ˈ_iː_ _s_ˈ_ɛ_d_ '
        '_ð_ə_ _h_ˈ_ɛ_n_ _ɹ_ˈ_ʌ_f_l_ɪ_ŋ_ _h_ɜː_ _f_ˈ_ɛ_ð_ɚ_z_ _æ_n_d_ _ð_ˈ_ɛ_n_ _ʃ_ˈ_eɪ_k_ɪ_ŋ_ _'
        'ð_ˌ_ɛ_m_ _ˌ_ɪ_n_t_ʊ_ _p_l_ˈ_eɪ_s',
        's_k_ˈ_ɜː_ɹ_ɪ_ɪ_ŋ_ _f_ɔːɹ_ _m_ˈ_eɪ_dʒ_ɚ_ _s_t_ˈ_oː_ɹ_ɪ_z_ _w_ɛ_n_ˌ_ɛ_v_ɚ_ɹ_ _'
        'æ_n_d_ _w_ɛɹ_ɹ_ˈ_ɛ_v_ɚ_ _ð_eɪ_ _k_ʊ_d_ _b_iː_ _f_ˈ_aʊ_n_d',
        'ˈ_æ_k_ʃ_ə_n_z_ _d_ˈ_ɑː_t_ _s_æ_m_ˈ_ɪ_ɹ_ _ˈ_ɛ_m_ _b_ˈ_ɑː_b_uː_ _ɪ_z_ _ɐ_ _p_ɹ_ə_f_ˈ_'
        'ɛ_s_ɚ_ _h_ˌ_uː_ _ɹ_ˈ_oʊ_t_ _ɐ_n_ _ˈ_ɑːɹ_ɾ_ɪ_k_əl_ _ɐ_b_ˌ_aʊ_t_ _k_l_ˈ_æ_s_ɹ_uː_m_ _'
        'k_l_ˈ_aɪ_m_ə_t_ _æ_n_d_ _s_ˈ_oʊ_ʃ_əl_ _ɪ_n_t_ˈ_ɛ_l_ɪ_dʒ_ə_n_s',
        'k_ˈ_ɑː_p_i_ _b_ˈ_aɪ',
        'f_ɔː_ɹ_ _ɛ_ɡ_z_ˈ_æ_m_p_əl',
        'ɔːɹ_ _s_ˈ_ʌ_m_t_aɪ_m_z_ _b_ˌ_iː_ɪ_ŋ_ _ɪ_l_ˈ_ɪ_m_ᵻ_n_ˌ_eɪ_ɾ_ᵻ_d',
        """ʌ_v_ _f_ˈ_aɪ_v_ _s_t_ˈ_eɪ_dʒ_ᵻ_z_
_ _ˈ_aɪ_ _p_ɹ_ˌ_ɛ_p_ɚ_ɹ_ˈ_eɪ_ʃ_ə_n_
_ _ɹ_ˌ_oʊ_m_ə_n_ _t_ˈ_uː_ _ˌ_ɪ_n_k_j_uː_b_ˈ_eɪ_ʃ_ə_n_
_ _ɹ_ˌ_oʊ_m_ə_n_ _θ_ɹ_ˈ_iː_ _ˌ_ɪ_n_t_ɪ_m_ˈ_eɪ_ʃ_ə_n_
_ _ɹ_ˌ_oʊ_m_ə_n_ _f_ˈ_oːɹ_ _ɪ_l_ˌ_uː_m_ᵻ_n_ˈ_eɪ_ʃ_ə_n""",
        'aɪ_ _h_ˈ_ɑː_ _θ_ˈ_ɔː_t_ _t_ˈ_ɪ_l_ _m_aɪ_ _b_ɹ_ˈ_eɪ_n_z_ _ˈ_eɪ_k_t_b_ɪ_l_i_ _m_ˌ_iː_ '
        '_dʒ_ˈ_ɑː_n_ _aɪ_ _h_ˈ_æ_v_ _ɐ_n_ _aɪ_ _'
        's_ˈ_eɪ_ _ɐ_ɡ_ˈ_ɛ_n_ _ð_ɚ_z_ _n_ˈ_oʊ_ _h_ˈ_ɛ_l_p_ _f_ɔː_ɹ_ _ˌ_ʌ_s_ _b_ˌ_ʌ_t_ _h_ˌ_æ_v_ɪ_ŋ_ '
        '_f_ˈ_eɪ_θ_ _ˈ_aɪ_ _ð_ə_ _'
        'j_ˈ_uː_n_iə_n_ _θ_ˈ_eɪ_l_ _w_ˈ_ɪ_n_ _ð_ə_ _d_ˈ_eɪ_ _s_ˈ_iː_ _ɪ_f_ _ð_eɪ_ _d_ˈ_ʌ_n_ɑː_t',
        'ð_ə_ _f_ˈ_ʊ_t_b_ɔː_l_ _p_l_ˈ_eɪ_ _ʌ_v_ð_ə_ _d_ˈ_ɛ_k_eɪ_d_ _ɔːɹ_ _ð_ə_ _k_ˈ_ɑː_n_s_ɜː_t_ _'
        'ə_v_ə_ _l_ˈ_aɪ_f_t_aɪ_m_\n_\n_'
        'ð_eɪ_l_ _s_p_ˈ_ɛ_n_d_ _w_ˈ_ʌ_n_p_ˈ_ɔɪ_n_t_f_ˈ_aɪ_v_ _b_ˈ_ɪ_l_iə_n_ _d_ˈ_ɑː_l_ɚ_z_ _'
        'ˌ_ɑː_n_ _t_w_ˈ_ɛ_n_t_i_ˈ_eɪ_t_ _θ_ˈ_aʊ_z_ə_n_d_ _ɪ_v_ˈ_ɛ_n_t_s_ _ɹ_ˈ_eɪ_n_dʒ_ɪ_ŋ_ _'
        'f_ɹ_ʌ_m_ _b_ɹ_ˈ_ɔː_d_w_eɪ_ _t_ə_ _s_ˈ_uː_p_ɚ_ _b_ˈ_oʊ_l_z',
        'f_ˈ_ɔːɹ_tʃ_ə_n_ə_t_l_i_ _k_l_ˈ_ʌ_b_ _m_ˈ_ɛ_d_ _h_ɐ_z_ _ɡ_ˈ_ɪ_v_ə_n_ _ˌ_ʌ_s_ _ɐ_n_ '
        '_ˈ_æ_n_t_ɪ_d_ˌ_oʊ_t_\n_\n_'
        'ð_ə_ _k_l_ˈ_ʌ_b_ _m_ˈ_ɛ_d_ _v_eɪ_k_ˈ_eɪ_ʃ_ə_n_ _v_ˈ_ɪ_l_ɪ_dʒ_ _w_ˌ_ɛ_ɹ_ _ˈ_ɔː_l_ _ð_oʊ_z_ '
        '_p_ɹ_ˈ_aɪ_m_ _'
        'd_ɪ_s_t_ˈ_ɜː_b_ɚ_z_ _ʌ_v_ð_ə_ _p_ˈ_iː_s_ _l_ˈ_aɪ_k_ _t_ˈ_ɛ_l_ɪ_f_ˌ_oʊ_n_z_ _k_l_ˈ_ɑː_k_s_'
        ' _æ_n_d_ _n_ˈ_uː_z_p_eɪ_p_ɚ_z_ _ɑːɹ_ _ɡ_ˈ_ɔ_n',
        'ɔːɹ_ _f_ˈ_ɔː_t_ _ˌ_oʊ_v_ɚ_ _ð_ɪ_s_ _ɡ_ɹ_ˈ_aʊ_n_d_\n_'
        'f_ɔːɹ_ _ð_ɪ_s_ _k_ə_m_j_ˈ_uː_n_ɪ_ɾ_i_ _ɐ_n_ _ɔːɹ_d_ˈ_iə_l_ _ð_æ_t_ _s_t_ˈ_ɑːɹ_ɾ_ᵻ_d_ '
        '_w_ɪ_ð_ _ə_f_ˈ_ɛ_n_s_ _ʌ_n_s_ˈ_ɜː_t_ə_n_t_i_ _æ_n_d_ _ˈ_aʊ_t_ɹ_eɪ_dʒ_ _ˈ_ɛ_n_d_ᵻ_d_ '
        '_ɐ_m_ˈ_ɪ_d_s_t_ _h_ˈ_ɔː_ɹ_ɚ_ _p_ˈ_ɑː_v_ɚ_ɾ_i',
        'ɐ_ _b_ˈ_æ_n_d_ _k_ˈ_ɑː_l_ɚ_ _ʃ_ˈ_ɜː_t_ _b_ˈ_ʌ_ʔ_n̩_d_ _t_ˈ_aɪ_t_ _ɐ_ɹ_ˈ_aʊ_n_d_ _ð_ə_ '
        '_θ_ɹ_ˈ_oʊ_t_ _æ_n_d_ _ɐ_ _d_ˈ_ɑːɹ_k_ _b_ˈ_ɪ_z_n_ə_s_ _dʒ_ˈ_æ_k_ɪ_t_\n_\n_'
        'h_iː_ _p_ˈ_oʊ_z_d_ _ð_ə_ _k_ˈ_ʌ_p_əl_ _b_ˈ_oːɹ_d_s_t_ˈ_ɪ_f_ _ɪ_n_ _f_ɹ_ˈ_ʌ_n_t_ _ə_v_ə_ '
        '_p_l_ˈ_eɪ_n_ _h_ˈ_aʊ_s_\n_\n_ð_ə_ _m_ˈ_æ_n',
    ]

    for g, p in zip(grapheme, phoneme):
        assert _grapheme_to_phoneme(g, separator='_') == p


def test__grapheme_to_phoneme_perserve_punctuation():
    assert """ʌ_v_ _f_ˈ_aɪ_v_ _s_t_ˈ_eɪ_dʒ_ᵻ_z_:_
_(_ˈ_aɪ_)_ _p_ɹ_ˌ_ɛ_p_ɚ_ɹ_ˈ_eɪ_ʃ_ə_n_,_
_(_ɹ_ˌ_oʊ_m_ə_n_ _t_ˈ_uː_)_ _ˌ_ɪ_n_k_j_uː_b_ˈ_eɪ_ʃ_ə_n_,_
_(_ɹ_ˌ_oʊ_m_ə_n_ _θ_ɹ_ˈ_iː_)_ _ˌ_ɪ_n_t_ɪ_m_ˈ_eɪ_ʃ_ə_n_,_
_(_ɹ_ˌ_oʊ_m_ə_n_ _f_ˈ_oːɹ_)_ _ɪ_l_ˌ_uː_m_ᵻ_n_ˈ_eɪ_ʃ_ə_n""" == (
        _grapheme_to_phoneme_perserve_punctuation(
            """of 5 stages:
(i) preparation,
(ii) incubation,
(iii) intimation,
(iv) illumination""",
            separator='_'))

    assert "j_uː_ɹ_ˈ_iː_k_ɐ_ _w_ˈ_ɔː_k_s_ _ɑː_n_ð_ɪ_ _ˈ_ɛ_ɹ_ _ˈ_ɔː_l_ _ɹ_ˈ_aɪ_t_." == (
        _grapheme_to_phoneme_perserve_punctuation(
            "Eureka walks on the air all right.", separator='_'))

    assert " _ _h_ə_l_ˈ_oʊ_ _w_ˈ_ɜː_l_d_ _ " == (
        _grapheme_to_phoneme_perserve_punctuation("  Hello World  ", separator='_'))

    assert " _\n_\n_ _h_ə_l_ˈ_oʊ_ _w_ˈ_ɜː_l_d_ _\n_\n_ " == (
        _grapheme_to_phoneme_perserve_punctuation(" \n\n Hello World \n\n ", separator='_'))

    assert " _\n_\t_ _h_ə_l_ˈ_oʊ_ _w_ˈ_ɜː_l_d_ _\n_\t_ " == (
        _grapheme_to_phoneme_perserve_punctuation(" \n\t Hello World \n\t ", separator='_'))

    # NOTE: Test a number of string literals, see: https://docs.python.org/2.0/ref/strings.html
    # TODO: Investigate why tab characters are not preserved.
    assert " _\n_ _t_ˈ_ɛ_s_t_ _t_ˈ_ɛ_s_t_ _t_ˈ_ɛ_s_t_ _t_ˈ_ɛ_s_t_ _t_ˈ_ɛ_s_t_ _t_ˈ_ɛ_s_t_ " == (
        _grapheme_to_phoneme_perserve_punctuation(
            " \n test \t test \r test \v test \f test \a test \b ", separator='_'))


def test__grapheme_to_phoneme_perserve_punctuation__spacy_failure_cases():
    grapheme = [
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
    ]

    phoneme = [
        "d_ˈ_ɑː_t_ _s_æ_m_ˈ_ɪɹ_ _ˈ_ɛ_m_ _b_ˈ_ɑː_b_uː_ _ɪ_z_ _ɐ_ _p_ɹ_ə_f_ˈ_ɛ_s_ɚ_ _h_ˌ_uː_ "
        "_ɹ_ˈ_oʊ_t_ _ɐ_n_ _ˈ_ɑːɹ_ɾ_ɪ_k_əl_ _ɐ_b_ˌ_aʊ_t_ _k_l_ˈ_æ_s_ɹ_uː_m_ _k_l_ˈ_aɪ_m_ə_t_ "
        "_æ_n_d_ _s_ˈ_oʊ_ʃ_əl_ _ɪ_n_t_ˈ_ɛ_l_ɪ_dʒ_ə_n_s_.",
        "aɪ_ _h_ˈ_ɑː_ _θ_ˈ_ɔː_t_ _t_ˈ_ɪ_l_ _m_aɪ_ _b_ɹ_ˈ_eɪ_n_z_ _ˈ_eɪ_k_t_b_ɪ_l_i_ _m_ˌ_iː_,_ _"
        "dʒ_ˈ_ɑː_n_,_ _aɪ_ _h_ˈ_æ_v_._ _ɐ_n_ _aɪ_ _s_ˈ_eɪ_ _ɐ_ɡ_ˈ_ɛ_n_,_ _ð_ɚ_z_ _n_ˈ_oʊ_ "
        "_h_ˈ_ɛ_l_p_ _f_ɔː_ɹ_ _ˌ_ʌ_s_ _b_ˌ_ʌ_t_ _h_ˌ_æ_v_ɪ_ŋ_ _f_ˈ_eɪ_θ_ _ˈ_aɪ_ "
        "_ð_ə_ _j_ˈ_uː_n_iə_n_._ _θ_ˈ_eɪ_l_ _w_ˈ_ɪ_n_ _ð_ə_ _d_ˈ_eɪ_,_ _s_ˈ_iː_ _ɪ_f_ _ð_eɪ_ "
        "_d_ˈ_ʌ_n_ɑː_t_!",
        "w_ɪɹ_ _ɡ_ˈ_ɛ_t_n_'_ _ɐ_ _l_ˈ_ɑː_ŋ_ _w_ˈ_eɪ_ _f_ɹ_ʌ_m_ _h_ˈ_oʊ_m_._ _æ_n_d_ _s_ˈ_iː_ _"
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
        "_ɔː_s_p_ˈ_ɪ_ʃ_ə_s_n_ə_s_,_ _v_ˈ_ɪ_ɡ_ɚ_ _s_l_ˈ_æ_ʃ_ _v_aɪ_t_ˈ_æ_l_ɪ_ɾ_i_,_ _ˈ_æ_n_d_,_ "
        "_s_p_ə_s_ˈ_ɪ_f_ɪ_k_l_i_,_ _s_ˈ_ɛ_k_ʃ_uː_əl_ _ɐ_l_ˈ_ʊɹ_ _ɔːɹ_ _tʃ_ˈ_ɑːɹ_m_ "
        "_–_ _ˈ_ɔː_l_ _ʌ_v_w_ˈ_ɪ_tʃ_ _ɑːɹ_ _n_ˌ_ɑː_t_ _ˈ_oʊ_n_l_i_ _ɐ_s_k_ɹ_ˈ_aɪ_b_d_ _ɪ_n_ "
        "_t_ˈ_ɛ_k_s_t_,_ _b_ˌ_ʌ_t_ _ˈ_iː_k_w_əl_i_ _t_ə_b_i_ _ɹ_ˈ_ɛ_d_ _ɪ_n_ _ˈ_ɪ_m_ɪ_dʒ_ɹ_i_.",
    ]

    for g, p in zip(grapheme, phoneme):
        assert _grapheme_to_phoneme_perserve_punctuation(g, separator='_') == p


def test_input_encoder():
    encoder = InputEncoder(['a', 'b', 'c'], [JUDY_BIEBER, MARY_ANN])
    encoded = encoder.batch_encode([('a', JUDY_BIEBER)])[0]
    assert encoder.decode(encoded) == ('ˈ|eɪ', JUDY_BIEBER)


def test_input_encoder__reversible():
    encoder = InputEncoder(['a', 'b', 'c'], [JUDY_BIEBER, MARY_ANN])

    with pytest.raises(ValueError):  # Text is not reversible
        encoder.encode(('d', JUDY_BIEBER))

    with pytest.raises(ValueError):  # Speaker is not reversible
        encoder.encode(('a', HILARY_NORIEGA))

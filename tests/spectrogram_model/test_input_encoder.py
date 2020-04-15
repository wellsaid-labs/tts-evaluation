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

    assert get_result('Hello world') == 'h_ə_l_ˈoʊ w_ˈɜː_l_d'
    assert get_result('How are you?') == 'h_ˈaʊ ɑːɹ j_uː?'
    assert get_result('I\'m great!') == 'aɪ_m ɡ_ɹ_ˈeɪ_t!'


def test__grapheme_to_phoneme__separator():
    # Test using `service_separator`.
    assert _grapheme_to_phoneme('Hello World', separator='_') == 'h_ə_l_ˈoʊ w_ˈɜː_l_d'

    with pytest.raises(AssertionError):  # Test separator is not unique
        _grapheme_to_phoneme('Hello World', separator='ə')


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
        '  h_ə_l_ˈoʊ w_ˈɜː_l_d  ',
        'h_ə_l_ˈoʊ w_ˈɜː_l_d  ',
        '  h_ə_l_ˈoʊ w_ˈɜː_l_d',
        ' \n h_ə_l_ˈoʊ w_ˈɜː_l_d \n ',
        ' \n\n h_ə_l_ˈoʊ w_ˈɜː_l_d \n\n ',
        'ɪ_t ɪ_z k_ˈɑː_m_ə_n_l_i ˈɑːɹ_ɡ_j_uː_d ð_æ_t_ð_ə n_ˈoʊ_ʃ_ə_n ʌ_v',
        'æ_n_d t_ɹ_ˈɑː_t f_ˈɛ_d ɪ_t ɐ h_ˈæ_n_d_f_əl ʌ_v f_ɹ_ˈɛ_ʃ b_l_ˈuː '
        'k_l_ˈoʊ_v_ɚ æ_n_d s_m_ˈuː_ð_d æ_n_d p_ˈɛ_ɾ_ᵻ_d ɪ_t ʌ_n_t_ˈɪ_l ð_ə '
        'l_ˈæ_m w_ʌ_z ˈiː_ɡ_ɚ t_ə f_ˈɑː_l_oʊ h_ɜː w_ɛɹ_ɹ_ˈɛ_v_ɚ ʃ_iː m_ˌaɪ_t '
        'ɡ_ˈoʊ',
        'ð_ə h_ˈæ_b_ɪ_t_s ʌ_v m_ˈaɪ_n_d ð_æ_t k_ˈæ_ɹ_ɪ_k_t_ɚ_ɹ_ˌaɪ_z ɐ '
        'p_ˈɜː_s_ə_n s_t_ɹ_ˈɔ_ŋ_l_i d_ɪ_s_p_ˈoʊ_z_d t_ʊ_w_ˈɔːɹ_d '
        'k_ɹ_ˈɪ_ɾ_ɪ_k_əl θ_ˈɪ_ŋ_k_ɪ_ŋ ɪ_n_k_l_ˈuː_d ɐ d_ɪ_z_ˈaɪɚ t_ə '
        'f_ˈɑː_l_oʊ ɹ_ˈiː_z_ə_n æ_n_d ˈɛ_v_ɪ_d_ə_n_s w_ɛɹ_ɹ_ˈɛ_v_ɚ ð_eɪ '
        'm_ˈeɪ l_ˈiː_d',
        'ð_ə h_ˈæ_b_ɪ_t_s ʌ_v m_ˈaɪ_n_d ð_æ_t k_ˈæ_ɹ_ɪ_k_t_ɚ_ɹ_ˌaɪ_z ɐ '
        'p_ˈɜː_s_ə_n s_t_ɹ_ˈɔ_ŋ_l_i d_ɪ_s_p_ˈoʊ_z_d t_ʊ_w_ˈɔːɹ_d '
        'k_ɹ_ˈɪ_ɾ_ɪ_k_əl θ_ˈɪ_ŋ_k_ɪ_ŋ ɪ_n_k_l_ˈuː_d ɐ d_ɪ_z_ˈaɪɚ t_ə '
        'f_ˈɑː_l_oʊ ɹ_ˈiː_z_ə_n æ_n_d ˈɛ_v_ɪ_d_ə_n_s w_ɛɹ_ɹ_ˈɛ_v_ɚ '
        'ð_eɪ m_ˈeɪ l_ˈiː_d ɐ s_ˌɪ_s_t_ə_m_ˈæ_ɾ_ɪ_k',
        'b_ˌʌ_t w_ɛɹ_ɹ_ˈɛ_v_ɚ ð_eɪ f_ˈɔː_t ɪ_n n_ˈɔːɹ_θ ˈæ_f_ɹ_ɪ_k_ə ɔːɹ '
        'ð_ə s_ˈaʊ_θ p_ɐ_s_ˈɪ_f_ɪ_k ɔːɹ w_ˈɛ_s_t_ɚ_n j_ˈʊ_ɹ_ə_p ð_ɪ '
        'ˈɪ_n_f_ə_n_t_ɹ_i b_ˈoːɹ ð_ə b_ɹ_ˈʌ_n_t ʌ_v_ð_ə f_ˈaɪ_ɾ_ɪ_ŋ '
        'ɑː_n_ð_ə ɡ_ɹ_ˈaʊ_n_d æ_n_d s_ˈɛ_v_ə_n ˌaʊ_ɾ_ə_v t_ˈɛ_n s_ˈʌ_f_ɚ_d '
        'k_ˈæ_ʒ_uː_əl_ɾ_ɪ_z',
        'aɪ l_ˈeɪ ˈɛ_ɡ_z w_ɛɹ_ɹ_ˈɛ_v_ɚ_ɹ aɪ h_ˈæ_p_ə_n t_ə b_ˈiː s_ˈɛ_d ð_ə '
        'h_ˈɛ_n ɹ_ˈʌ_f_l_ɪ_ŋ h_ɜː f_ˈɛ_ð_ɚ_z æ_n_d ð_ˈɛ_n ʃ_ˈeɪ_k_ɪ_ŋ '
        'ð_ˌɛ_m ˌɪ_n_t_ʊ p_l_ˈeɪ_s',
        's_k_ˈɜː_ɹ_ɪ_ɪ_ŋ f_ɔːɹ m_ˈeɪ_dʒ_ɚ s_t_ˈoː_ɹ_ɪ_z w_ɛ_n_ˌɛ_v_ɚ_ɹ '
        'æ_n_d w_ɛɹ_ɹ_ˈɛ_v_ɚ ð_eɪ k_ʊ_d b_iː f_ˈaʊ_n_d',
        'ˈæ_k_ʃ_ə_n_z d_ˈɑː_t s_æ_m_ˈɪ_ɹ ˈɛ_m b_ˈɑː_b_uː ɪ_z ɐ p_ɹ_ə_f_ˈ'
        'ɛ_s_ɚ h_ˌuː ɹ_ˈoʊ_t ɐ_n ˈɑːɹ_ɾ_ɪ_k_əl ɐ_b_ˌaʊ_t k_l_ˈæ_s_ɹ_uː_m '
        'k_l_ˈaɪ_m_ə_t æ_n_d s_ˈoʊ_ʃ_əl ɪ_n_t_ˈɛ_l_ɪ_dʒ_ə_n_s',
        'k_ˈɑː_p_i b_ˈaɪ',
        'f_ɔː_ɹ ɛ_ɡ_z_ˈæ_m_p_əl',
        'ɔːɹ s_ˈʌ_m_t_aɪ_m_z b_ˌiː_ɪ_ŋ ɪ_l_ˈɪ_m_ᵻ_n_ˌeɪ_ɾ_ᵻ_d',
        """ʌ_v f_ˈaɪ_v s_t_ˈeɪ_dʒ_ᵻ_z
 ˈaɪ p_ɹ_ˌɛ_p_ɚ_ɹ_ˈeɪ_ʃ_ə_n
 ɹ_ˌoʊ_m_ə_n t_ˈuː ˌɪ_n_k_j_uː_b_ˈeɪ_ʃ_ə_n
 ɹ_ˌoʊ_m_ə_n θ_ɹ_ˈiː ˌɪ_n_t_ɪ_m_ˈeɪ_ʃ_ə_n
 ɹ_ˌoʊ_m_ə_n f_ˈoːɹ ɪ_l_ˌuː_m_ᵻ_n_ˈeɪ_ʃ_ə_n""",
        'aɪ h_ˈɑː θ_ˈɔː_t t_ˈɪ_l m_aɪ b_ɹ_ˈeɪ_n_z ˈeɪ_k_t_b_ɪ_l_i m_ˌiː dʒ_ˈɑː_n aɪ h_ˈæ_v ɐ_n aɪ '
        's_ˈeɪ ɐ_ɡ_ˈɛ_n ð_ɚ_z n_ˈoʊ h_ˈɛ_l_p f_ɔː_ɹ ˌʌ_s b_ˌʌ_t h_ˌæ_v_ɪ_ŋ f_ˈeɪ_θ ˈaɪ ð_ə '
        'j_ˈuː_n_iə_n θ_ˈeɪ_l w_ˈɪ_n ð_ə d_ˈeɪ s_ˈiː ɪ_f ð_eɪ d_ˈʌ_n_ɑː_t',
        'ð_ə f_ˈʊ_t_b_ɔː_l p_l_ˈeɪ ʌ_v_ð_ə d_ˈɛ_k_eɪ_d ɔːɹ ð_ə k_ˈɑː_n_s_ɜː_t '
        'ə_v_ə l_ˈaɪ_f_t_aɪ_m\n\n'
        'ð_eɪ_l s_p_ˈɛ_n_d w_ˈʌ_n_p_ˈɔɪ_n_t_f_ˈaɪ_v b_ˈɪ_l_iə_n d_ˈɑː_l_ɚ_z '
        'ˌɑː_n t_w_ˈɛ_n_t_i_ˈeɪ_t θ_ˈaʊ_z_ə_n_d ɪ_v_ˈɛ_n_t_s ɹ_ˈeɪ_n_dʒ_ɪ_ŋ '
        'f_ɹ_ʌ_m b_ɹ_ˈɔː_d_w_eɪ t_ə s_ˈuː_p_ɚ b_ˈoʊ_l_z',
        'f_ˈɔːɹ_tʃ_ə_n_ə_t_l_i k_l_ˈʌ_b m_ˈɛ_d h_ɐ_z ɡ_ˈɪ_v_ə_n ˌʌ_s ɐ_n ˈæ_n_t_ɪ_d_ˌoʊ_t\n\n'
        'ð_ə k_l_ˈʌ_b m_ˈɛ_d v_eɪ_k_ˈeɪ_ʃ_ə_n v_ˈɪ_l_ɪ_dʒ w_ˌɛ_ɹ ˈɔː_l ð_oʊ_z p_ɹ_ˈaɪ_m '
        'd_ɪ_s_t_ˈɜː_b_ɚ_z ʌ_v_ð_ə p_ˈiː_s l_ˈaɪ_k t_ˈɛ_l_ɪ_f_ˌoʊ_n_z k_l_ˈɑː_k_s æ_n_d '
        'n_ˈuː_z_p_eɪ_p_ɚ_z ɑːɹ ɡ_ˈɔ_n',
        'ɔːɹ f_ˈɔː_t ˌoʊ_v_ɚ ð_ɪ_s ɡ_ɹ_ˈaʊ_n_d\n'
        'f_ɔːɹ ð_ɪ_s k_ə_m_j_ˈuː_n_ɪ_ɾ_i ɐ_n ɔːɹ_d_ˈiə_l ð_æ_t s_t_ˈɑːɹ_ɾ_ᵻ_d w_ɪ_ð ə_f_ˈɛ_n_s '
        'ʌ_n_s_ˈɜː_t_ə_n_t_i æ_n_d ˈaʊ_t_ɹ_eɪ_dʒ ˈɛ_n_d_ᵻ_d ɐ_m_ˈɪ_d_s_t h_ˈɔː_ɹ_ɚ p_ˈɑː_v_ɚ_ɾ_i',
        'ɐ b_ˈæ_n_d k_ˈɑː_l_ɚ ʃ_ˈɜː_t b_ˈʌ_ʔ_n̩_d t_ˈaɪ_t ɐ_ɹ_ˈaʊ_n_d ð_ə θ_ɹ_ˈoʊ_t æ_n_d ɐ '
        'd_ˈɑːɹ_k b_ˈɪ_z_n_ə_s dʒ_ˈæ_k_ɪ_t\n\n'
        'h_iː p_ˈoʊ_z_d ð_ə k_ˈʌ_p_əl b_ˈoːɹ_d_s_t_ˈɪ_f ɪ_n f_ɹ_ˈʌ_n_t ə_v_ə p_l_ˈeɪ_n h_ˈaʊ_s\n\n'
        'ð_ə m_ˈæ_n',
    ]

    for g, p in zip(grapheme, phoneme):
        assert _grapheme_to_phoneme(g, separator='_') == p


def test__grapheme_to_phoneme_perserve_punctuation():
    assert """ʌ_v f_ˈaɪ_v s_t_ˈeɪ_dʒ_ᵻ_z:
(ˈaɪ) p_ɹ_ˌɛ_p_ɚ_ɹ_ˈeɪ_ʃ_ə_n,
(ɹ_ˌoʊ_m_ə_n t_ˈuː) ˌɪ_n_k_j_uː_b_ˈeɪ_ʃ_ə_n,
(ɹ_ˌoʊ_m_ə_n θ_ɹ_ˈiː) ˌɪ_n_t_ɪ_m_ˈeɪ_ʃ_ə_n,
(ɹ_ˌoʊ_m_ə_n f_ˈoːɹ) ɪ_l_ˌuː_m_ᵻ_n_ˈeɪ_ʃ_ə_n""" == _grapheme_to_phoneme_perserve_punctuation(
        """of 5 stages:
(i) preparation,
(ii) incubation,
(iii) intimation,
(iv) illumination""",
        separator='_')

    assert "j_uː_ɹ_ˈiː_k_ɐ w_ˈɔː_k_s ɑː_n_ð_ɪ ˈɛ_ɹ ˈɔː_l ɹ_ˈaɪ_t." == (
        _grapheme_to_phoneme_perserve_punctuation(
            "Eureka walks on the air all right.", separator='_'))

    assert "  h_ə_l_ˈoʊ w_ˈɜː_l_d  " == (
        _grapheme_to_phoneme_perserve_punctuation("  Hello World  ", separator='_'))

    assert " \n\n h_ə_l_ˈoʊ w_ˈɜː_l_d \n\n " == (
        _grapheme_to_phoneme_perserve_punctuation(" \n\n Hello World \n\n ", separator='_'))


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
        "d_ˈɑː_t s_æ_m_ˈɪɹ ˈɛ_m b_ˈɑː_b_uː ɪ_z ɐ p_ɹ_ə_f_ˈɛ_s_ɚ h_ˌuː "
        "ɹ_ˈoʊ_t ɐ_n ˈɑːɹ_ɾ_ɪ_k_əl ɐ_b_ˌaʊ_t k_l_ˈæ_s_ɹ_uː_m k_l_ˈaɪ_m_ə_t æ_n_d "
        "s_ˈoʊ_ʃ_əl ɪ_n_t_ˈɛ_l_ɪ_dʒ_ə_n_s.",
        "aɪ h_ˈɑː θ_ˈɔː_t t_ˈɪ_l m_aɪ b_ɹ_ˈeɪ_n_z ˈeɪ_k_t_b_ɪ_l_i m_ˌiː, "
        "dʒ_ˈɑː_n, aɪ h_ˈæ_v. ɐ_n aɪ s_ˈeɪ ɐ_ɡ_ˈɛ_n, ð_ɚ_z n_ˈoʊ h_ˈɛ_l_p f_ɔː_ɹ "
        "ˌʌ_s b_ˌʌ_t h_ˌæ_v_ɪ_ŋ f_ˈeɪ_θ ˈaɪ ð_ə j_ˈuː_n_iə_n. θ_ˈeɪ_l w_ˈɪ_n ð_ə d_ˈeɪ, "
        "s_ˈiː ɪ_f ð_eɪ d_ˈʌ_n_ɑː_t!",
        "w_ɪɹ ɡ_ˈɛ_t_n' ɐ l_ˈɑː_ŋ w_ˈeɪ f_ɹ_ʌ_m h_ˈoʊ_m. æ_n_d s_ˈiː "
        "h_ˌaʊ ð_ə k_l_ˈaʊ_d_z ɑːɹ ɹ_ˈoʊ_l_ɪ_ŋ dʒ_ˈʌ_s_t ə_b_ˈʌ_v ˌʌ_s, "
        "ɹ_ɪ_m_ˈɑːɹ_k_t ð_ə b_ˈɔɪ, h_ˌuː w_ʌ_z ˈɔː_l_m_oʊ_s_t "
        "æ_z ʌ_n_ˈiː_z_i æ_z k_ˈæ_p_t_ɪ_n b_ˈɪ_l.",
        "aɪ aɪ d_ˈoʊ_n_t ˈɛ_s-ˈɛ_s-s_ˈiː ˌɛ_n_i-θ_ˈɪ_ŋ "
        "f_ˈʌ_n_i b_ˈaʊ_t ɪ_t! h_iː s_t_ˈæ_m_ɚ_d.",
        "b_ˌʌ_t d_ˈoʊ_n_t w_ˈʌ_ɹ_i, ð_ɛ_ɹ_ˌɑːɹ p_l_ˈɛ_n_t_i ʌ_v "
        "t_ˈɔɪ_z ð_æ_t ɑːɹ s_ˈeɪ_f--æ_n_d f_ˈʌ_n--f_ɔːɹ j_ʊɹ tʃ_ˈaɪ_l_d.",
        "ˌɪ_n_d_ˈiː_d, f_ɚ_ð_ə ɹ_ˈɔɪ_əl b_ˈɑː_d_i, ɐ ɹ_ˈæ_ð_ɚ_ɹ ʌ_n_j_ˈuː_ʒ_uː_əl "
        "s_ˈɛ_t ʌ_v aɪ_d_ˈiə_l ˈæ_t_ɹ_ɪ_b_j_ˌuː_t_s ɪ_m_ˈɜː_dʒ_ᵻ_z ɪ_n_ð_ə "
        "m_ˌɛ_s_ə_p_ə_t_ˈeɪ_m_iə_n l_ˈɛ_k_s_ɪ_k_ə_n: ɐ_n ɐ_k_j_ˌuː_m_j_ʊ_l_ˈeɪ_ʃ_ə_n ʌ_v ɡ_ˈʊ_d "
        "f_ˈɔːɹ_m ɔːɹ b_ɹ_ˈiː_d_ɪ_ŋ, ɔː_s_p_ˈɪ_ʃ_ə_s_n_ə_s, v_ˈɪ_ɡ_ɚ s_l_ˈæ_ʃ v_aɪ_t_ˈæ_l_ɪ_ɾ_i, "
        "ˈæ_n_d, s_p_ə_s_ˈɪ_f_ɪ_k_l_i, s_ˈɛ_k_ʃ_uː_əl ɐ_l_ˈʊɹ ɔːɹ tʃ_ˈɑːɹ_m – ˈɔː_l ʌ_v_w_ˈɪ_tʃ "
        "ɑːɹ n_ˌɑː_t ˈoʊ_n_l_i ɐ_s_k_ɹ_ˈaɪ_b_d ɪ_n t_ˈɛ_k_s_t, b_ˌʌ_t ˈiː_k_w_əl_i t_ə_b_i "
        "ɹ_ˈɛ_d ɪ_n ˈɪ_m_ɪ_dʒ_ɹ_i.",
    ]

    for g, p in zip(grapheme, phoneme):
        assert _grapheme_to_phoneme_perserve_punctuation(g, separator='_') == p


def test_input_encoder():
    encoder = InputEncoder(['a', 'b', 'c'], [JUDY_BIEBER, MARY_ANN])
    encoded = encoder.batch_encode([('a', JUDY_BIEBER)])[0]
    assert encoder.decode(encoded) == ('ˈeɪ', JUDY_BIEBER)


def test_input_encoder__reversible():
    encoder = InputEncoder(['a', 'b', 'c'], [JUDY_BIEBER, MARY_ANN])

    with pytest.raises(ValueError):  # Text is not reversible
        encoder.encode(('d', JUDY_BIEBER))

    with pytest.raises(ValueError):  # Speaker is not reversible
        encoder.encode(('a', HILARY_NORIEGA))

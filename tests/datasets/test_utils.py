import spacy
import torch

from itertools import product

from src.datasets import Gender
from src.datasets import JACK_RUTKOWSKI
from src.datasets import Speaker
from src.datasets import TextSpeechRow
from src.datasets.utils import add_predicted_spectrogram_column
from src.datasets.utils import add_spectrogram_column
from src.datasets.utils import filter_
from src.datasets.utils import normalize_text_clean_punctuation
from src.datasets.utils import phonemize_data
from src.datasets.utils import _phonemize_text
from src.datasets.utils import _separator_token
from tests._utils import get_tts_mocks


def test_filter_():
    a = TextSpeechRow(text='this is a test', speaker=Speaker('Stay', Gender.FEMALE), audio_path='')
    b = TextSpeechRow(text='this is a test', speaker=Speaker('Stay', Gender.FEMALE), audio_path='')
    c = TextSpeechRow(
        text='this is a test', speaker=Speaker('Remove', Gender.FEMALE), audio_path='')

    assert filter_(lambda e: e.speaker != Speaker('Remove', Gender.FEMALE), [a, b, c]) == [a, b]


def test_add_predicted_spectrogram_column():
    mocks = get_tts_mocks(add_spectrogram=True)
    dataset = mocks['dev_dataset']

    # In memory
    processed = add_predicted_spectrogram_column(
        dataset, mocks['spectrogram_model_checkpoint'], mocks['device'], 1, on_disk=False)
    assert len(processed) == len(dataset)
    assert all(torch.is_tensor(r.predicted_spectrogram) for r in processed)

    # On disk
    processed = add_predicted_spectrogram_column(
        dataset, mocks['spectrogram_model_checkpoint'], mocks['device'], 1, on_disk=True)
    assert len(processed) == len(dataset)
    assert all(r.predicted_spectrogram.path.exists() for r in processed)
    assert len(set(r.predicted_spectrogram.path for r in processed)) == len(dataset)
    assert all(r.audio_path.stem in r.predicted_spectrogram.path.stem for r in processed)

    # On disk and cached from the previous execution
    cached = add_predicted_spectrogram_column(
        dataset, mocks['spectrogram_model_checkpoint'], mocks['device'], 1, on_disk=True)
    assert processed == cached

    # No audio path
    dataset = [r._replace(audio_path=None) for r in dataset]
    processed = add_predicted_spectrogram_column(
        dataset, mocks['spectrogram_model_checkpoint'], mocks['device'], 1, on_disk=True)
    assert len(processed) == len(dataset)
    assert all(r.predicted_spectrogram.path.exists() for r in processed)
    assert len(set(r.predicted_spectrogram.path for r in processed)) == len(dataset)


def test_add_spectrogram_column():
    mocks = get_tts_mocks()

    # In memory
    processed = add_spectrogram_column(mocks['dev_dataset'], on_disk=False)
    assert all(torch.is_tensor(r.spectrogram) for r in processed)
    assert all(torch.is_tensor(r.spectrogram_audio) for r in processed)

    # On disk
    processed = add_spectrogram_column(mocks['dev_dataset'], on_disk=True)
    assert all(r.audio_path.stem in r.spectrogram.path.stem for r in processed)

    # On disk and cached from the previous execution
    cached = add_spectrogram_column(mocks['dev_dataset'], on_disk=True)
    assert cached == processed


def test_normalize_text_clean_punctuation():
    text = [
        'What do you know."Management',
        'actions .Sameer M Babu is a professor who wrote an article about classroom climate and social intelligence.',
        'Movies are purely for entertainment, while eLearning is more serious. It\'s ... well ... learning.  No entertainment there!'
    ]
    expected = [
        'What do you know." Management',
        'actions. Sameer M Babu is a professor who wrote an article about classroom climate and social intelligence.',
        'Movies are purely for entertainment, while eLearning is more serious. It\'s ... well ... learning.  No entertainment there!'
    ]

    dataset = [TextSpeechRow(t, s, None) for t, s in product(text, [JACK_RUTKOWSKI])]
    normalized = normalize_text_clean_punctuation(dataset)

    assert all(e in [n.text for n in normalized] for e in expected)


nlp = spacy.load('en_core_web_sm')  # Must load small EN core to use in testing.


def test_phonemize_data():
    mocks = get_tts_mocks()
    dataset = mocks['dataset']

    curr_text = [r.text for r in dataset]
    phonemized = phonemize_data(dataset, nlp)

    for i, p in enumerate(phonemized):
        assert curr_text[i] != p.text

    text = 'Eureka walks on the air all right.'
    phonemes = 'jPHONE_SEPARATORuːPHONE_SEPARATORɹPHONE_SEPARATORiːPHONE_SEPARATORkPHONE_SEPARATORɐ wPHONE_SEPARATORɔːPHONE_SEPARATORkPHONE_SEPARATORs ɑːPHONE_SEPARATORnPHONE_SEPARATORðPHONE_SEPARATORɪ ɛPHONE_SEPARATORɹ ɔːPHONE_SEPARATORl ɹPHONE_SEPARATORaɪPHONE_SEPARATORt.'
    phonemized_text = [p.text for p in phonemized]

    assert text in curr_text
    assert phonemes in phonemized_text


def test_phonemize_data__assert_separator_token():
    """ Test `_phonemize_phrases` when `_separator_token` is not unique from the original text in
    the dataset.
    """

    text = "This is some sample" + _separator_token + "text."
    dataset = [TextSpeechRow(t, s, None) for t, s in product([text], [JACK_RUTKOWSKI])]

    try:
        phonemize_data(dataset, nlp)
    except AssertionError:
        assert True


def test_phonemize_text__spaces__punctuation():
    """ Test `_phonemize_phrases` and `_phonemize_text` on strange spaces and punctuation from
    some of our datasets.
    """
    text = """  of 5 stages:

(i) preparation,
 (ii) incubation,
(iii) intimation,
(iv) illumination"""
    expected = [('phrase_to_phonemize', '  of 5 stages'), ':\n\n(',
                ('phrase_to_phonemize', 'i'), ') ', ('phrase_to_phonemize', 'preparation'), ',\n (',
                ('phrase_to_phonemize', 'ii'), ') ', ('phrase_to_phonemize', 'incubation'), ',\n(',
                ('phrase_to_phonemize', 'iii'), ') ', ('phrase_to_phonemize', 'intimation'), ',\n(',
                ('phrase_to_phonemize', 'iv'), ') ', ('phrase_to_phonemize', 'illumination')]
    assert expected == _phonemize_text(text, nlp)

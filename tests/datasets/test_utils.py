import spacy
import torch

from src.datasets import Gender
from src.datasets import Speaker
from src.datasets import TextSpeechRow
from src.datasets.utils import add_predicted_spectrogram_column
from src.datasets.utils import add_spectrogram_column
from src.datasets.utils import filter_
from src.datasets.utils import phonemize_data
from src.datasets.utils import _phonemize_text
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


nlp = spacy.load('en_core_web_sm')  # Must load small EN core to use in testing.


def test_phonemize_data():
    mocks = get_tts_mocks()
    dataset = mocks['dataset']

    curr_text = [r.text for r in dataset]
    phonemized = phonemize_data(dataset, nlp)

    for i, p in enumerate(phonemized):
        assert curr_text[i] != p.text

    text = 'Eureka walks on the air all right.'
    phonemes = 'juːɹiːkɐ wɔːks ɑːnðɪ ɛɹ ɔːl ɹaɪt.'
    phonemized_text = [p.text for p in phonemized]

    assert text in curr_text
    assert phonemes in phonemized_text


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

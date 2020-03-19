from itertools import groupby

import logging
import re
import subprocess

from torchnlp.encoders import Encoder
from torchnlp.encoders import LabelEncoder
from torchnlp.encoders.text import CharacterEncoder

import spacy

from src.utils import disk_cache

logger = logging.getLogger(__name__)


@disk_cache
def _grapheme_to_phoneme(grapheme,
                         service='espeak',
                         flags=('--ipa=3', '-q', '-ven-us', '--stdin'),
                         separator=None,
                         service_separator='_',
                         split_new_line=True):
    """ Convert graphemes into phonemes without perserving punctuation.

    Args:
        grapheme (str): The graphemes to convert to phonemes.
        service (str, optional): The service used to compute phonemes.
        flags (list of str, optional): The list of flags to add to the service.
        separator (None or str, optional): The separator used to seperate phonemes.
        service_separator (str, optional): The separator used by the service between phonemes.
        strip (bool, optional): `espeak` can be inconsistent in it's handling of outer spacing;
            therefore, it's recommended both the `espeak` output and input is trimmed.
        split_new_line (bool, optioanl): `espeak` can be inconsistent in it's handling of new lines;
            therefore, it's recommended to split the input on new lines.

    Returns:
        phoneme (str)
    """
    if split_new_line:
        splits = grapheme.split('\n')
        if len(splits) > 1:
            return '\n'.join([
                _grapheme_to_phoneme(
                    s,
                    service=service,
                    flags=flags,
                    separator=separator,
                    service_separator=service_separator,
                    split_new_line=split_new_line) for s in splits
            ])

    # NOTE: `espeak` can be inconsistent in it's handling of outer spacing; therefore, it's
    # recommended both the `espeak` output and input is trimmed.
    original = grapheme
    grapheme = grapheme.rstrip()
    stripped_right = original[len(grapheme):]
    grapheme = grapheme.lstrip()
    stripped_left = original[:len(original) - len(stripped_right) - len(grapheme)]

    # NOTE: The `--sep` flag is not supported by older versions of `espeak`.
    # NOTE: We recommend using `--stdin` otherwise `espeak` might misinterpret an input like
    # "--For this community," as a flag.
    phoneme = subprocess.check_output(
        'echo "%s" | %s %s' % (grapheme, service, ' '.join(flags)), shell=True).decode('utf-8')
    assert separator is None or separator not in phoneme, 'The separator is not unique.'

    phoneme = phoneme.strip()
    phoneme = ' '.join([s.strip() for s in phoneme.split('\n')]) if split_new_line else phoneme

    # NOTE: Replace multiple separators in a row without any phonemes in between with one separator.
    phoneme = re.sub(r'%s+' % service_separator, service_separator, phoneme)
    phoneme = re.sub(r'%s+\s+' % service_separator, ' ', phoneme)

    phoneme = stripped_left + phoneme + stripped_right
    return phoneme if separator is None else phoneme.replace(service_separator, separator)


@disk_cache
def _grapheme_to_phoneme_perserve_punctuation(text, nlp):
    """ Convert grapheme to phoneme while perserving punctuation.

    Args:
        text (str): Graphemes.
        nlp (Language): spaCy object used to tokenize and POS tag text.

    Returns:
        (str): Phonemes with the original punctuation.
    """
    tokens = nlp(text)

    assert len(tokens) > 0, 'Zero tokens were found in text: %s' % text
    assert text == ''.join(t.text_with_ws for t in tokens), 'Detokenization failed: %s' % text

    phrases = []
    is_punctuation = []
    for is_punct, group in groupby(tokens, lambda t: t.is_punct):
        phrases.append(''.join([t.text_with_ws for t in group]))
        is_alpha_numeric = any(c.isalpha() or c.isdigit() for c in phrases[-1])
        if is_punct and is_alpha_numeric:
            logger.warning('Punctuation contains alphanumeric characters: %s' % phrases[-1])
        is_punctuation.append(is_punct and not is_alpha_numeric)

    return ''.join([
        p if is_punct else _grapheme_to_phoneme(p) for p, is_punct in zip(phrases, is_punctuation)
    ])


class InputEncoder(Encoder):
    """ Handles encoding and decoding input to the spectrogram model.

    Args:
        text_samples (list of str): Examples used to make the text encoder.
        speaker_samples (list of src.datasets.constants.Speaker): Examples used to make the speaker
          encoder.
        spacy_model (str)
    """

    def __init__(self, text_samples, speaker_samples, spacy_model='en_core_web_sm', **kwargs):
        super().__init__(**kwargs)

        self.text_encoder = CharacterEncoder(text_samples, enforce_reversible=True)
        self.speaker_encoder = LabelEncoder(
            speaker_samples, reserved_labels=[], enforce_reversible=True)
        self.nlp = spacy.load(spacy_model, disable=['parser', 'ner'])

    def encode(self, object_):
        """
        Args:
            object_ (tuple): (
              text (str)
              speaker (src.datasets.constants.Speaker)
            )

        Returns:
            (torch.Tensor [num_tokens]): Encoded text.
            (torch.Tensor [1]): Encoded speaker.
        """
        return (
            self.text_encoder.encode(
                _grapheme_to_phoneme_perserve_punctuation(object_[0], self.nlp)),
            self.speaker_encoder.encode(object_[1]).view(1),
        )

    def decode(self, encoded):
        """
        Args:
            encoded (tuple): (
              text (torch.Tensor [num_tokens]): Encoded text.
              speaker (torch.Tensor [1]): Encoded speaker.
            )

        Returns:
            text (str)
            speaker (src.datasets.constants.Speaker)
        """
        return self.text_encoder.decode(encoded[0]), self.speaker_encoder.decode(
            encoded[1].squeeze())

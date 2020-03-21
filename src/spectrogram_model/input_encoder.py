from itertools import groupby
from multiprocessing.pool import ThreadPool

import logging
import os
import re
import shlex
import subprocess

from torchnlp.encoders import Encoder
from torchnlp.encoders import LabelEncoder
from torchnlp.encoders.text import CharacterEncoder
from tqdm import tqdm

import spacy

from src.environment import IS_TESTING_ENVIRONMENT
from src.utils import disk_cache
from src.utils import strip

logger = logging.getLogger(__name__)


def _grapheme_to_phoneme(grapheme, **kwargs):
    """ Convert graphemes into phonemes without perserving punctuation.

    NOTE: `grapheme` is split on new lines because `espeak` is inconsistent in it's handling of new
    lines.

    Args:
        grapheme (str): The graphemes to convert to phonemes.
        service (str, optional): The service used to compute phonemes.
        flags (list of str, optional): The list of flags to add to the service.
        separator (None or str, optional): The separator used to seperate phonemes.
        service_separator (str, optional): The separator used by the service between phonemes.

    Returns:
        phoneme (str)
    """
    return '\n'.join([_grapheme_to_phoneme_helper(s, **kwargs) for s in grapheme.split('\n')])


@disk_cache
def _grapheme_to_phoneme_helper(grapheme,
                                service='espeak',
                                flags=('--ipa=3', '-q', '-ven-us', '--stdin'),
                                separator=None,
                                service_separator='_'):
    # NOTE: `espeak` can be inconsistent in it's handling of outer spacing; therefore, it's
    # recommended both the `espeak` output and input is trimmed.
    grapheme, stripped_left, stripped_right = strip(grapheme)

    # NOTE: The `--sep` flag is not supported by older versions of `espeak`.
    # NOTE: We recommend using `--stdin` otherwise `espeak` might misinterpret an input like
    # "--For this community," as a flag.
    command = 'echo %s | %s %s' % (shlex.quote(grapheme), service, ' '.join(flags))
    phoneme = subprocess.check_output(command, shell=True).decode('utf-8')
    assert not separator or separator == service_separator or separator not in phoneme, (
        'The separator is not unique.')

    phoneme = ' '.join([s.strip() for s in phoneme.strip().split('\n')])

    # NOTE: Replace multiple separators in a row without any phonemes in between with one separator.
    phoneme = re.sub(r'%s+' % service_separator, service_separator, phoneme)
    phoneme = re.sub(r'%s+\s+' % service_separator, ' ', phoneme)
    phoneme = phoneme.strip()

    phoneme = stripped_left + phoneme + stripped_right
    return phoneme if separator is None else phoneme.replace(service_separator, separator)


@disk_cache
def _grapheme_to_phoneme_perserve_punctuation(text, nlp, **kwargs):
    """ Convert grapheme to phoneme while perserving punctuation.

    Args:
        text (str): Graphemes.
        nlp (Language): spaCy object used to tokenize and POS tag text.

    Returns:
        (str): Phonemes with the original punctuation (as defined by spaCy).
    """
    tokens = nlp(text)

    assert len(tokens) > 0, 'Zero tokens were found in text: %s' % text
    assert text == ''.join(t.text_with_ws for t in tokens), 'Detokenization failed: %s' % text

    return_ = ''
    for is_punct, group in groupby(tokens, lambda t: t.is_punct):
        phrase = ''.join([t.text_with_ws for t in group])
        is_alpha_numeric = any(c.isalpha() or c.isdigit() for c in phrase)
        if is_punct and is_alpha_numeric:
            logger.warning('Punctuation contains alphanumeric characters: %s' % phrase)
        is_punct = is_punct and not is_alpha_numeric
        return_ += phrase if is_punct else _grapheme_to_phoneme(phrase, **kwargs)
    return return_


class InputEncoder(Encoder):
    """ Handles encoding and decoding input to the spectrogram model.

    Args:
        text_samples (list of str): Examples used to make the text encoder.
        speaker_samples (list of src.datasets.constants.Speaker): Examples used to make the speaker
          encoder.
        spacy_model (str): spaCy model to load for tokenization and POS tagging.
        chunksize (int): `chunksize` parameter passed to `imap`.
    """

    def __init__(self,
                 text_samples,
                 speaker_samples,
                 spacy_model='en_core_web_sm',
                 chunksize=128,
                 **kwargs):
        super().__init__(**kwargs)

        self.nlp = spacy.load(spacy_model, disable=['parser', 'ner'])

        with ThreadPool(1 if IS_TESTING_ENVIRONMENT else os.cpu_count()) as pool:
            length = len(text_samples)
            text_samples = pool.imap(self._preprocess, text_samples, chunksize=chunksize)
            text_samples = list(tqdm(text_samples, total=length))

        self.text_encoder = CharacterEncoder(text_samples, enforce_reversible=True)
        self.speaker_encoder = LabelEncoder(
            speaker_samples, reserved_labels=[], enforce_reversible=True)

    def _preprocess(self, text):
        # NOTE: The seperator is '' so that it's not included in the encoding.
        return _grapheme_to_phoneme_perserve_punctuation(
            text.strip(), nlp=self.nlp, separator='').strip()

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
            self.text_encoder.encode(self._preprocess(object_[0])),
            self.speaker_encoder.encode(object_[1]).view(1),
        )

    def decode(self, encoded):
        """
        NOTE: There is backward no grapheme to phoneme conversion.

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

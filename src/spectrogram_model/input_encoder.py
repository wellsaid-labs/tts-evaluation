from itertools import chain
from itertools import groupby

import re

from torchnlp.encoders import Encoder
from torchnlp.encoders import LabelEncoder
from torchnlp.encoders.text.static_tokenizer_encoder import StaticTokenizerEncoder

from src.datasets.utils import _separator_token


class InputEncoder(Encoder):
    """ Handles encoding and decoding input to the spectrogram model.

    Args:
        text_samples (list of str): Examples used to make the text encoder.
        speaker_samples (list of src.datasets.constants.Speaker): Examples used to make the speaker
          encoder.
    """

    def __init__(self, text_samples, speaker_samples, **kwargs):
        super().__init__(**kwargs)

        self.text_encoder = PhonesEncoder(text_samples, enforce_reversible=True)
        self.speaker_encoder = LabelEncoder(
            speaker_samples, reserved_labels=[], enforce_reversible=True)

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
        return self.text_encoder.encode(object_[0]), self.speaker_encoder.encode(object_[1]).view(1)

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


class PhonesEncoder(StaticTokenizerEncoder):
    """ Encodes text into a tensor containing punctuation characters, whitespace characters, and
    phoneme syllables. Phoneme syllables can be single or multiple characters.

    Args:
        **args: Arguments passed onto ``StaticTokenizerEncoder.__init__``.
        **kwargs: Keyword arguments passed onto ``StaticTokenizerEncoder.__init__``.
    """

    def __init__(self, text_samples, *args, **kwargs):
        all_punctuation = chain([re.findall(r'\W+', t) for t in text_samples])
        self._punctuation = set([p for punct in all_punctuation for p in punct])

        if 'tokenize' in kwargs:
            raise TypeError('``PhonesEncoder`` does not take keyword argument ``tokenize``.')

        if 'detokenize' in kwargs:
            raise TypeError('``PhonesEncoder`` does not take keyword argument ``detokenize``.')

        super().__init__(
            text_samples, *args, tokenize=self._tokenize, detokenize=self._detokenize, **kwargs)

    def _tokenize(self, s):
        """ Separates string into list of white spaces, punctuation characters, and phoneme
        syllables. Phone syllables can be single or multiple characters.

        Args:
            s (str): string, written in phonemic form with syllables separated by
            _separator_token

        Returns:
            list of str: list of string elements

        EXAMPLE:
        A phonemic transcription of a sentence will contain words, syllables, and spaces.
        Words will be separated by spaces and syllables within each word will be separted by
        the _separator_token. For example:

        PHRASE:   'Let's see the pigs.'

        PHONES:   'lTOKENˈɛTOKENtTOKENs sTOKENˈiː ðTOKENə pTOKENˈɪTOKENɡTOKENz'

        RETURN: ['l', 'ˈɛ', 't', 's', ' ', 's', 'ˈiː', ' ', 'ð', 'ə', ' ', 'p', 'ˈɪ', 'g', 'z']

        """

        tokens = []

        # Group by whitespace characters, punctuation characters, and remaining phoneme
        # characters representing whole words
        for (contains_spaces, contains_punctuation), word_characters in groupby(
                s, lambda c: (c.isspace(), c in self._punctuation)):
            if contains_spaces:
                # ``word_characters`` could contain any number of white spaces; append them each
                tokens.extend([c for c in word_characters])

            else:
                text = ''.join([c for c in word_characters])

                # Append each punctuation character or split phoneme word on separator token
                # and append each phoneme syllable
                tokens.extend(list(text) if contains_punctuation else text.split(_separator_token))

        unexpected = ['For', 'Itɛ', 'and', 'gett', 'promote']
        for u in unexpected:
            if u in tokens:
                print('############# FOUND UNEXPECTED TOKEN!\t%s\t%s' % (u, s))
        return tokens

    def _detokenize(self, s):
        """ Concatenates list elements back to original text including white spaces, punctuation
        characters, and phoneme syllables separated by the ``_separator_token``

        Args:
            s (list of str): list, containing string elements of white space characters,
            punctuation characters, and phonemic syllable characters

        Returns:
            (str): string in its original form

        EXAMPLE:
        INPUT: ['l', 'ˈɛ', 't', 's', ' ', 's', 'ˈiː', ' ', 'ð', 'ə', ' ', 'p', 'ˈɪ', 'g', 'z']

        PHONES:   lTOKENˈɛTOKENtTOKENs sTOKENˈiː ðTOKENə pTOKENˈɪTOKENɡTOKENz'

        RETURN:   'Let's see the pigs.'

        """
        detokenized = ''
        # Group by whitespace characters, punctuation characters, and remaining phoneme
        # characters representing whole words
        for (contains_spaces, contains_punctuation), word_characters in groupby(
                s, lambda c: (c.isspace(), c in self._punctuation)):
            if contains_spaces:
                # Concatenate all whitespace characters
                detokenized += ''.join([c for c in word_characters])

            else:
                # Concatenate all punctuation characters; or join all phoneme characters by
                # the ``_separator_token``` and concatenate the result
                detokenized += ('' if contains_punctuation else _separator_token).join(
                    [c for c in word_characters])

        return detokenized

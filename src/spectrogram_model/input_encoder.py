import logging
import re

import unidecode

from hparams import configurable
from hparams import HParam
from torchnlp.encoders import Encoder
from torchnlp.encoders import LabelEncoder
from torchnlp.encoders.text import DelimiterEncoder

from src.text import grapheme_to_phoneme_perserve_punctuation

logger = logging.getLogger(__name__)


class InputEncoder(Encoder):
    """ Handles encoding and decoding input to the spectrogram model.

    Args:
        text_samples (list of str): Examples used to make the text encoder.
        speaker_samples (list of src.datasets.constants.Speaker): Examples used to make the speaker
          encoder.
        **args: Additional arguments passed to `super()`.
        delimiter (string): A unique character used to tokenize text.
        **kwargs: Additional key-word arguments passed to `super()`.
    """

    @configurable
    def __init__(self, text_samples, speaker_samples, *args, delimiter=HParam(), **kwargs):
        super().__init__(*args, **kwargs)
        self.delimiter = delimiter
        self.text_encoder = DelimiterEncoder(
            delimiter, [self._preprocess(t) for t in text_samples], enforce_reversible=True)
        self.speaker_encoder = LabelEncoder(
            speaker_samples, reserved_labels=[], enforce_reversible=True)

    def _preprocess(self, text):
        text = text.strip()
        text = unidecode.unidecode(text)
        # Learn more:
        # https://en.wikipedia.org/wiki/Control_character
        # https://stackoverflow.com/questions/14946109/how-to-remove-escape-sequence-like-xe2-or-x0c-in-python
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff]', '', text)
        if self.delimiter and self.delimiter in text:
            raise ValueError('Text cannot contain these characters: %s' % self.delimiter)
        text = grapheme_to_phoneme_perserve_punctuation(text, separator=self.delimiter)
        return text

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
        preprocessed = self._preprocess(object_[0])

        try:
            return (self.text_encoder.encode(preprocessed),
                    self.speaker_encoder.encode(object_[1]).view(1))
        except ValueError:
            pass

        # NOTE: This allows us to ignore the earlier traceback. The earlier traceback is not
        # helpful in situations where the text is very long.
        difference = set(self.text_encoder.tokenize(preprocessed)).difference(
            set(self.text_encoder.vocab))
        difference = ', '.join(sorted(list(difference)))
        raise ValueError('Text cannot contain these characters: %s' % difference)

    def decode(self, encoded):
        """
        NOTE: There is no reverse operation for grapheme to phoneme conversion.

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

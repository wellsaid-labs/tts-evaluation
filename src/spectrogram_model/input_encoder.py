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


class InvalidTextValueError(ValueError):
    pass


class InvalidSpeakerValueError(ValueError):
    pass


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

    def _preprocess(self, text, **kwargs):
        # NOTE: Remove all ASCII control characters from 0 to 31 except `\t` and `\n`, see:
        # https://en.wikipedia.org/wiki/Control_character
        preprocessed = re.sub(r'[\x00-\x08\x0b\x0c\x0d\x0e-\x1f]', '', text)
        preprocessed = preprocessed.replace('\t', '  ')
        preprocessed = preprocessed.strip()
        if len(preprocessed) == 0:
            raise InvalidTextValueError('Text cannot be empty.')
        preprocessed = unidecode.unidecode(preprocessed)
        if self.delimiter and self.delimiter in preprocessed:
            raise InvalidTextValueError('Text cannot contain these characters: %s' % self.delimiter)
        preprocessed = grapheme_to_phoneme_perserve_punctuation(
            preprocessed, separator=self.delimiter, **kwargs)
        if len(preprocessed) == 0:
            raise InvalidTextValueError('Invalid text: "%s"' % text)
        return preprocessed

    def encode(self, object_, **kwargs):
        """
        Args:
            object_ (tuple): (
              text (str)
              speaker (src.datasets.constants.Speaker)
            )
            **kwargs: Additional keyword arguments passed onto `self._preprocess`.

        Returns:
            (torch.Tensor [num_tokens]): Encoded text.
            (torch.Tensor [1]): Encoded speaker.
        """
        preprocessed = self._preprocess(object_[0], **kwargs)

        try:
            encoded_text = self.text_encoder.encode(preprocessed)
        except ValueError:
            difference = set(self.text_encoder.tokenize(preprocessed)).difference(
                set(self.text_encoder.vocab))
            difference = ', '.join([repr(c)[1:-1] for c in sorted(list(difference))])
            raise InvalidTextValueError('Text cannot contain these characters: %s' % difference)

        try:
            encoded_speaker = self.speaker_encoder.encode(object_[1]).view(1)
        except ValueError:
            # NOTE: We do not expose speaker information in the `ValueError` because this error
            # is passed on to the public via the API.
            raise InvalidSpeakerValueError('Speaker is not available.')

        return encoded_text, encoded_speaker

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

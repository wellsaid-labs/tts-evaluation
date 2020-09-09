import logging
import typing

from torchnlp.encoders import Encoder
from torchnlp.encoders import LabelEncoder
from torchnlp.encoders.text import CharacterEncoder
from torchnlp.encoders.text import DelimiterEncoder

import torch

from src.datasets import Speaker

logger = logging.getLogger(__name__)


class InputEncoder(Encoder):
    """ Handles encoding and decoding input to the spectrogram model.

    Args:
        graphemes (list of str)
        phonemes (list of str)
        phoneme_seperator (str): Deliminator to split phonemes.
        speakers (list of src.datasets.Speaker)
        **args: Additional arguments passed to `super()`.
        **kwargs: Additional key-word arguments passed to `super()`.
    """

    _CASE_LABELS = ['upper', 'lower', 'other']

    def __init__(self, graphemes: typing.List[str], phonemes: typing.List[str],
                 phoneme_seperator: str, speakers: typing.List[Speaker], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.grapheme_encoder = CharacterEncoder([g.lower() for g in graphemes],
                                                 enforce_reversible=True)
        self.phoneme_seperator = phoneme_seperator
        self.phoneme_encoder = DelimiterEncoder(
            phoneme_seperator, phonemes, enforce_reversible=True)
        self.case_encoder = LabelEncoder(
            self._CASE_LABELS, reserved_labels=[], enforce_reversible=True)
        self.speaker_encoder = LabelEncoder(speakers, reserved_labels=[], enforce_reversible=True)

    def _get_case(self, c: str):
        if c.isupper():
            return self._CASE_LABELS[0]
        return self._CASE_LABELS[1] if c.islower() else self._CASE_LABELS[2]

    def encode(self, object_: typing.Tuple[str, str, Speaker],
               **kwargs) -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            object_ (tuple): (
              graphemes
              phonemes
              speaker
            )

        Returns:
            (torch.LongTensor [num_graphemes]): Encoded graphemes.
            (torch.LongTensor [num_graphemes]): Encoded letter cases.
            (list of torch.LongTensor [num_phonemes]): Encoded phonemes.
            (torch.LongTensor [1]): Encoded speaker.
        """
        assert len(object_[0]) > 0, 'Graphemes cannot be empty.'
        assert len(object_[1]) > 0, 'Phonemes cannot be empty.'

        return (
            self.grapheme_encoder.encode(object_[0].lower()),
            self.case_encoder.batch_encode([self._get_case(c) for c in object_[0]]),
            self.phoneme_encoder.encode(p),
            self.speaker_encoder.encode(object_[2]).view(1),
        )

    def decode(
        self, encoded: typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> typing.Tuple[str, str, Speaker]:
        """
        Args:
            encoded (tuple): (
                (torch.LongTensor [num_graphemes]): Encoded graphemes.
                (torch.LongTensor [num_graphemes]): Encoded cases.
                (torch.LongTensor [num_phonemes]): Encoded phonemes.
                (torch.LongTensor [1]): Encoded speaker.
            )

        Returns:
            graphemes
            phonemes
            speaker
        """
        graphemes = self.text_encoder.decode(encoded[0])
        cases = self.case_encoder.decode(encoded[1])
        iterator = zip(graphemes, cases)
        return (
            ''.join([g.upper() if c == self._CASE_LABELS[0] else g for g, c in iterator]),
            self.phoneme_encoder.decode(encoded[2]),
            self.speaker_encoder.decode(encoded[3]).squeeze(),
        )

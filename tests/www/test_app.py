from unittest import mock

from torchnlp.text_encoders import CharacterEncoder
from torchnlp.text_encoders import IdentityEncoder

import torch

from src.www.app import _synthesize
from src.datasets import Speaker


@mock.patch('src.www.app.spectrogram_model_checkpoint')
@mock.patch('src.www.app.signal_model_checkpoint')
def test_synthesize(mock_signal_model_checkpoint, mock_spectrogram_model_checkpoint):
    mock_signal_model_checkpoint.model.infer.return_value = (torch.LongTensor(3).zero_(),
                                                             torch.LongTensor(3).zero_(), None)
    mock_spectrogram_model_checkpoint.model.infer.return_value = (None, torch.FloatTensor(3, 4, 5),
                                                                  None, None, None)
    mock_spectrogram_model_checkpoint.text_encoder = CharacterEncoder(['This is a test.'])
    mock_spectrogram_model_checkpoint.speaker_encoder = IdentityEncoder([Speaker.LINDA_JOHNSON])

    text = 'This is a test.'
    speaker = '1'
    is_high_fidelity = True
    return_ = _synthesize(text, speaker, is_high_fidelity)
    assert return_.is_file()
    return_.unlink()

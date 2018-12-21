from unittest import mock

from torchnlp.text_encoders import CharacterEncoder
from torchnlp.text_encoders import IdentityEncoder

import torch

from src.datasets import Speaker
from src.www.app import _synthesize


class MockModel(torch.nn.Module):

    def __init__(self, forward=None):
        super().__init__()
        self.forward = forward


@mock.patch('src.www.app.get_spectrogram_model_checkpoint')
@mock.patch('src.www.app.get_signal_model')
def test_synthesize(mock_get_signal_model, mock_get_spectrogram_model_checkpoint):
    mock_spectrogram_model_checkpoint = mock.Mock()
    mock_spectrogram_model_checkpoint.model = MockModel(
        lambda *args, **kwargs: (None, torch.FloatTensor(3, 4, 5), None, None, None))
    mock_spectrogram_model_checkpoint.text_encoder = CharacterEncoder(['This is a test.'])
    mock_spectrogram_model_checkpoint.speaker_encoder = IdentityEncoder([Speaker.LINDA_JOHNSON])
    mock_get_spectrogram_model_checkpoint.return_value = mock_spectrogram_model_checkpoint

    mock_get_signal_model.return_value = MockModel(
        lambda *args, **kwargs: (torch.LongTensor(3).zero_(), torch.LongTensor(3).zero_(), None))

    text = 'This is a test.'
    speaker = '1'
    is_high_fidelity = True
    return_ = _synthesize(text, speaker, is_high_fidelity)
    assert return_.is_file()
    return_.unlink()

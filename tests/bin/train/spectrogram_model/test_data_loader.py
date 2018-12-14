from torchnlp.text_encoders import CharacterEncoder
from torchnlp.text_encoders import IdentityEncoder

import torch

from src.bin.train.spectrogram_model.data_loader import DataLoader

from tests.utils import get_example_spectrogram_text_speech_rows


def test_data_loader():
    num_frames = [50, 100]
    data = get_example_spectrogram_text_speech_rows(num_frames=num_frames)
    text_encoder = CharacterEncoder([r.text for r in data])
    speaker_encoder = IdentityEncoder([r.speaker for r in data])
    batch_size = 2

    # Smoke test
    iterator = DataLoader(
        data,
        batch_size,
        torch.device('cpu'),
        text_encoder=text_encoder,
        speaker_encoder=speaker_encoder,
    )
    assert len(iterator) == 1
    item = next(iter(iterator))

    # Test collate
    assert item.spectrogram_mask[0].sum().item() == sum(num_frames)

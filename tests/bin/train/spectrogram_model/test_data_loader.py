import torch

from src.bin.train.spectrogram_model.data_loader import DataLoader
from src.spectrogram_model import InputEncoder

from tests.utils import get_example_spectrogram_text_speech_rows


def test_data_loader():
    num_frames = [50, 100]
    data = get_example_spectrogram_text_speech_rows(num_frames=num_frames)
    input_encoder = InputEncoder([r.text for r in data], [r.speaker for r in data])
    batch_size = 2

    # Smoke test
    iterator = DataLoader(
        data, batch_size, torch.device('cpu'), input_encoder=input_encoder, use_tqdm=True)
    assert len(iterator) == 1
    item = next(iter(iterator))

    # Test collate
    assert item.spectrogram_mask[0].sum().item() == sum(num_frames)

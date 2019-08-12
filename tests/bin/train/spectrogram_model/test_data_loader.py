from src.bin.train.spectrogram_model.data_loader import DataLoader

from tests._utils import get_tts_mocks


def test_data_loader():
    mocks = get_tts_mocks(add_spectrogram=True)
    data = mocks['dev_dataset']
    batch_size = 2

    # Smoke test
    iterator = DataLoader(data, batch_size, mocks['device'], input_encoder=mocks['input_encoder'])
    assert len(iterator) == len(data) // batch_size

    # Test collate
    total_frames = sum([r.spectrogram.shape[0] for r in data])
    print([r.spectrogram.shape[0] for r in data])
    print([r.spectrogram_mask[0].sum().item() for r in iterator])
    assert sum([r.spectrogram_mask[0].sum().item() for r in iterator]) == total_frames

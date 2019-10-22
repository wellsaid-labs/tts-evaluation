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
    samples = list(iterator)  # The iterator contains some randomness everytime it's sampled.
    assert sum([r.spectrogram_mask.tensor.sum().item() for r in samples]) == (
        sum([r.spectrogram[1].sum().item() for r in samples]))
    assert sum([r.spectrogram_mask.tensor.sum().item() for r in samples]) == (
        sum([r.spectrogram.tensor.sum(dim=2).nonzero().shape[0] for r in samples]))
    assert sum([r.spectrogram_expanded_mask[0].sum().item() for r in samples]) == (
        sum([r.spectrogram.tensor.nonzero().shape[0] for r in samples]))
    assert sum([r.stop_token.tensor.sum().item() for r in samples]) == len(samples) * batch_size

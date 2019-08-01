import torch

from src.utils.batch_predict_spectrograms import batch_predict_spectrograms
from tests._utils import get_tts_mocks


def test_batch_predict_spectrograms():
    mocks = get_tts_mocks()

    # Return predicted spectrogram in memory
    predictions = batch_predict_spectrograms(
        data=mocks['dev_dataset'],
        input_encoder=mocks['input_encoder'],
        model=mocks['spectrogram_model'],
        batch_size=1,
        device=mocks['device'],
        aligned=False)
    assert len(predictions) == len(mocks['dev_dataset'])
    assert torch.is_tensor(predictions[0])


def test_batch_predict_spectrograms__disk():
    """ Test the `filenames` parameter. """
    mocks = get_tts_mocks()

    # Return predicted spectrogram on disk
    filenames = ['/tmp/tensor_%d.npy' % d for d in range(len(mocks['dev_dataset']))]
    predictions = batch_predict_spectrograms(
        data=mocks['dev_dataset'],
        input_encoder=mocks['input_encoder'],
        model=mocks['spectrogram_model'],
        batch_size=1,
        device=mocks['device'],
        filenames=filenames,
        aligned=False)
    assert len(predictions) == len(mocks['dev_dataset'])
    # Ensure predictions are sorted in the right order
    for prediction, filename in zip(predictions, filenames):
        assert filename in str(prediction.path)


def test_batch_predict_spectrograms__aligned():
    """ Test the `aligned` parameter. """
    mocks = get_tts_mocks(add_spectrogram=True)
    dataset = mocks['dev_dataset']
    predictions = batch_predict_spectrograms(
        data=dataset,
        input_encoder=mocks['input_encoder'],
        model=mocks['spectrogram_model'],
        batch_size=1,
        device=mocks['device'],
        aligned=True)
    assert len(predictions) == len(dataset)
    assert all(torch.is_tensor(p) for p in predictions)

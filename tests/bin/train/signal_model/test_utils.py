import numpy as np
import torch

from src.bin.train.signal_model._utils import load_data
from src.bin.train.signal_model._utils import set_hparams


def test_load_data():
    train, dev = load_data(
        predicted_source='tests/_test_data/signal_dataset/',
        log_mel_spectrogram_prefix='log_mel_spectrogram',
        signal_prefix='signal',
        extension='.npy',
        predicted=True)
    assert len(train) == 1
    assert len(dev) == 1

    row = dev[0]
    assert row['slice']['input_signal'].shape[1] == 2
    assert row['slice']['input_signal'].shape[0] == row['slice']['target_signal_coarse'].shape[0]
    assert row['slice']['target_signal_fine'].shape[0] == row['slice']['target_signal_fine'].shape[
        0]
    assert row['signal'].shape[0] % row['log_mel_spectrogram'].shape[0] == 0

    # Test input signal and target signal are one timestep off
    expected = torch.stack(
        (row['slice']['target_signal_coarse'], row['slice']['target_signal_fine']), dim=1)[:-1]
    np.testing.assert_allclose(row['slice']['input_signal'][1:].numpy(), expected.numpy())


def test_set_hparams():
    """ Smoke test hparams are being applied """
    set_hparams()
    model = torch.nn.LSTM(10, 10)
    optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()))
    assert optimizer.defaults['eps'] == 10**-8

from src.bin.train.feature_model._utils import load_data
from src.bin.train.feature_model._utils import set_hparams


def test_set_hparams():
    # Smoke test
    set_hparams()


def test_load_data():
    train, dev, text_encoder = load_data(
        source_train='tests/_test_data/feature_dataset/train',
        source_dev='tests/_test_data/feature_dataset/dev')
    assert len(train) == 1
    assert len(dev) == 1
    assert text_encoder.decode(train[0]['text']) == 'Yup!'

    # Smoke test
    dev[0]

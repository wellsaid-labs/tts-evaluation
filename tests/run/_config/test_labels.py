from run._config import (
    Cadence,
    DatasetType,
    Label,
    get_config_label,
    get_dataset_label,
    get_model_label,
)
from run._config.labels import _label
from run.data._loader.english import SAM_SCHOLL


def test__label():
    """Test `run._config._label` handles recursive templates."""
    label = Label("512_spectrogram")
    assert _label("{name}", name="{fft_length}_spectrogram", fft_length=512) == label


def test_get_dataset_label():
    """Test `run._config.get_dataset_label` formats a label appropriately."""
    expected = Label("static/dataset/train/test")
    assert get_dataset_label("test", Cadence.STATIC, DatasetType.TRAIN) == expected
    expected = Label("static/dataset/dev/sam_scholl/test")
    result = get_dataset_label("test", Cadence.STATIC, DatasetType.DEV, SAM_SCHOLL)
    assert result == expected


def test_get_model_label():
    """Test `run._config.get_model_label` formats a label appropriately."""
    expected = Label("static/model/test")
    assert get_model_label("test", Cadence.STATIC) == expected
    expected = Label("static/model/sam_scholl/test")
    assert get_model_label("test", Cadence.STATIC, SAM_SCHOLL) == expected


def test_get_config_label():
    """Test `run._config.get_config_label` formats a label appropriately."""
    expected = Label("static/config/test")
    assert get_config_label("test", Cadence.STATIC) == expected
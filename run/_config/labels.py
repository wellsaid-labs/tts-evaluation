import enum
import logging
import typing

from run.data._loader import Speaker

logger = logging.getLogger(__name__)


class Cadence(enum.Enum):
    STEP: typing.Final = "step"
    MULTI_STEP: typing.Final = "multi_step"
    RUN: typing.Final = "run"  # NOTE: Measures statistic over the course of a "run"
    STATIC: typing.Final = "static"


class DatasetType(enum.Enum):
    TRAIN: typing.Final = "train"
    DEV: typing.Final = "dev"
    TEST: typing.Final = "test"


class Device(enum.Enum):
    CUDA: typing.Final = "cuda"
    CPU: typing.Final = "cpu"


Label = typing.NewType("Label", str)


class GetLabel(typing.Protocol):
    def __call__(self, **kwargs) -> Label:
        ...


def _label(template: str, *args, **kwargs) -> Label:
    """Format `template` recursively, and return a `Label`.

    TODO: For recursive formatting, don't reuse arguments.
    """
    while True:
        formatted = template.format(*args, **kwargs)
        if formatted == template:
            return Label(formatted)
        template = formatted


def _speaker(speaker: Speaker) -> str:
    """Get a unique label per speaker.

    TODO: For bilingual speakers, consider adding additional parameters, or annotating the
    text instead.
    """
    name = f"{speaker.label}/{speaker.style.value.lower().replace(' ', '_')}"
    name += "/post" if speaker.post else ""
    return name


def get_dataset_label(
    name: str,
    cadence: Cadence,
    type_: DatasetType,
    speaker: typing.Optional[Speaker] = None,
    **kwargs,
) -> Label:
    """Label something related to a dataset."""
    kwargs = dict(cadence=cadence.value, type=type_.value, name=name, **kwargs)
    if speaker is None:
        return _label("{cadence}/dataset/{type}/{name}", **kwargs)
    return _label("{cadence}/dataset/{type}/{speaker}/{name}", speaker=_speaker(speaker), **kwargs)


def get_model_label(
    name: str, cadence: Cadence, speaker: typing.Optional[Speaker] = None, **kwargs
) -> Label:
    """Label something related to the model."""
    kwargs = dict(cadence=cadence.value, name=name, **kwargs)
    if speaker is None:
        return _label("{cadence}/model/{name}", **kwargs)
    return _label("{cadence}/model/{speaker}/{name}", speaker=_speaker(speaker), **kwargs)


def get_signal_model_label(
    name: str, *args, fft_length: typing.Optional[int] = None, **kwargs
) -> Label:
    """Label something related to the signal model that allows `fft_length` to be specified."""
    fft_length_ = "multi" if fft_length is None else fft_length
    name = f"{fft_length_}_fft_length/{name}"
    return get_model_label(name, *args, **kwargs)


def get_config_label(name: str, cadence: Cadence = Cadence.STATIC, **kwargs) -> Label:
    """Label something related to a configuration."""
    return _label("{cadence}/config/{name}", cadence=cadence.value, name=name, **kwargs)


def get_environment_label(name: str, cadence: Cadence = Cadence.STATIC, **kwargs) -> Label:
    """Label something related to a environment."""
    return _label("{cadence}/environment/{name}", cadence=cadence.value, name=name, **kwargs)


def get_timer_label(
    name: str, device: Device = Device.CPU, cadence: Cadence = Cadence.STATIC, **kwargs
) -> Label:
    """Label something related to a performance."""
    template = "{cadence}/timer/{device}/{name}"
    return _label(template, cadence=cadence.value, device=device.value, name=name, **kwargs)

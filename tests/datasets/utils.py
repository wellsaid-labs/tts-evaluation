import urllib.request

from src.utils import Checkpoint
from src.datasets import Speaker


# Check the URL requested is valid
def urlretrieve_side_effect(url, *args, **kwargs):
    # TODO: Fix failure case if internet does not work
    assert urllib.request.urlopen(url).getcode() == 200


# Check the URL requested is valid
def _download_file_from_drive_side_effect(_, url, **kwargs):
    # TODO: Fix failure case if internet does not work
    assert urllib.request.urlopen(url).getcode() == 200


def compute_spectrogram_side_effect(audio_path, text, speaker, spectrogram_model_checkpoint):
    assert isinstance(text, str)
    assert isinstance(speaker, Speaker)
    assert isinstance(spectrogram_model_checkpoint, Checkpoint)
    return audio_path, 'spectrogram_path', 'predicted_spectrogram_path'

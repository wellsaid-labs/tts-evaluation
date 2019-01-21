from contextlib import contextmanager

import torch

from src.utils import OnDiskTensor
from src.datasets import SpectrogramTextSpeechRow
from src.datasets import Speaker


class MockCometML():

    def __init__(*args, **kwargs):
        pass

    @contextmanager
    def train(self, *args, **kwargs):
        yield self

    @contextmanager
    def validate(self, *args, **kwargs):
        yield self

    def __getattr__(self, attr):
        return lambda *args, **kwargs: self


class MockOnDiskTensor(OnDiskTensor):

    def __init__(self, path, tensor, does_exist=True):
        self.tensor = tensor
        self._does_exist = does_exist
        self.path = path
        self.allow_pickle = False

    @property
    def shape(self):
        return self.tensor.shape

    def to_tensor(self):
        return self.tensor

    def does_exist(self):
        return self._does_exist

    def unlink(self):
        return self

    def from_tensor(self, tensor):
        self.tensor = tensor
        return self


def get_example_spectrogram_text_speech_rows(samples_per_frame=2,
                                             frame_channels=80,
                                             num_frames=[50, 100]):
    tensor = lambda p, *a, **k: MockOnDiskTensor(p, torch.FloatTensor(*a, **k).fill_(0))
    return [
        SpectrogramTextSpeechRow(
            text='Hi, my name is Hilary.',
            speaker=Speaker.HILARY_NORIEGA,
            audio_path=None,
            spectrogram=tensor('spectrogram.npy', num_frames[0], frame_channels),
            spectrogram_audio=tensor('audio.npy', samples_per_frame * num_frames[0]),
            predicted_spectrogram=tensor('predicted.npy', num_frames[0], frame_channels),
            metadata=None),
        SpectrogramTextSpeechRow(
            text='Hi, my name is Linda.',
            speaker=Speaker.LINDA_JOHNSON,
            audio_path=None,
            spectrogram=tensor('spectrogram_2.npy', num_frames[1], frame_channels),
            spectrogram_audio=tensor('audio_2.npy', samples_per_frame * num_frames[1]),
            predicted_spectrogram=tensor('predicted_2.npy', num_frames[1], frame_channels),
            metadata=None)
    ]

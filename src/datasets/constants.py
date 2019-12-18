from enum import Enum
from collections import namedtuple

# Args:
#     text (str or torch.LongTensor): Text transcription of audio
#     speaker (Speaker or torch.LongTensor): Speaker speaking the text.
#     audio_path (Path or str): Path to the ``.wav`` file of text.
#     spectrogram (OnDiskTensor or torch.FloatTensor): Two-dimensional spectrogram of the audio.
#     spectrogram_audio (OnDiskTensor or torch.HalfTensor): One-dimensional signal aligned to the
#         spectrogram.
#     predicted_spectrogram (OnDiskTensor or torch.FloatTensor): Two-dimensional spectrogram of the
#         audio predicted by some model.
#     metadata (dict): Metadata for the data such as the data source.
TextSpeechRow = namedtuple('TextSpeechRow', [
    'text', 'speaker', 'audio_path', 'spectrogram', 'spectrogram_audio', 'predicted_spectrogram',
    'metadata'
])
TextSpeechRow.__new__.__defaults__ = (None, None, None, {})


class Gender(Enum):
    FEMALE = 0
    MALE = 1


class Speaker(object):

    def __init__(self, name, gender):
        self.name = name
        self.gender = gender

    def __eq__(self, other):
        if isinstance(other, Speaker):
            return self.name == other.name and self.gender == other.gender

        # Learn more:
        # https://stackoverflow.com/questions/878943/why-return-notimplemented-instead-of-raising-notimplementederror
        return NotImplemented

    def __hash__(self):
        # Learn more:
        # https://computinglife.wordpress.com/2008/11/20/why-do-hash-functions-use-prime-numbers/
        return 32 * hash(self.name) + 97 * hash(self.gender)

    def __repr__(self):
        return '%s(name=\'%s\', gender=%s)' % (self.__class__.__name__, self.name, self.gender.name)

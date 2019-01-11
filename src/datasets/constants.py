from enum import Enum
from collections import namedtuple

# Args:
#     text (str or torch.Tensor): Text transcription of audio
#     speaker (Speaker or torch.Tensor): Speaker speaking the text.
#     audio_path (Path or str): Path to the ``.wav`` file of text.
#     predicted_spectrogram (OnDiskTensor or torch.Tensor): Two-dimensional spectrogram of the audio
#         predicted by some model.
#     spectrogram (OnDiskTensor or torch.Tensor): Two-dimensional spectrogram of the audio.
#     spectrogram_audio (OnDiskTensor or torch.Tensor): One-dimensional signal aligned to the
#         spectrogram.
#     metadata (dict): Metadata for the data such as the data source.
TextSpeechRow = namedtuple('TextSpeechRow', ['text', 'speaker', 'audio_path', 'metadata'])
SpectrogramTextSpeechRow = namedtuple('SpectrogramTextSpeechRow', [
    'text', 'speaker', 'audio_path', 'spectrogram', 'spectrogram_audio', 'predicted_spectrogram',
    'metadata'
])


class Gender(Enum):
    FEMALE = 0
    MALE = 1


class _LengthMetaClass(type):

    def __len__(self):
        return self.class_length()


class Speaker(object, metaclass=_LengthMetaClass):

    def __init__(self, name, gender, id):
        self.name = name
        self.gender = gender
        self.id = id

    def __int__(self):
        return self.id

    def __eq__(self, other):
        return self.id == other.id

    def __hash__(self):
        return self.id

    @classmethod
    def class_length(class_):
        speakers = [v for n, v in vars(class_).items() if isinstance(v, Speaker)]
        return len(speakers)

    def __repr__(self):
        return '%s(name=\'%s\', gender=%s, id=%d)' % (self.__class__.__name__, self.name,
                                                      self.gender.name, self.id)


_speaker_args = {
    'JUDY_BIEBER': ('Judy Bieber', Gender.FEMALE),
    'MARY_ANN': ('Mary Ann', Gender.FEMALE),
    'ELLIOT_MILLER': ('Elliot Miller', Gender.MALE),
    'HILARY_NORIEGA': ('Hilary Noriega', Gender.FEMALE),
    'LINDA_JOHNSON': ('Linda Johnson', Gender.FEMALE)
}
for id, (key, args) in enumerate(_speaker_args.items()):
    speaker = Speaker(*args, id=id)
    setattr(Speaker, key, speaker)
    setattr(Speaker, str(id), speaker)

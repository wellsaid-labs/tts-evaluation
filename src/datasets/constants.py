from enum import Enum


class Gender(Enum):
    FEMALE = 0
    MALE = 1


class _LengthMetaClass(type):

    def __len__(self):
        return self.class_length()


class Speaker(object, metaclass=_LengthMetaClass):

    def __init__(self, name, gender, index):
        self.name = name
        self.gender = gender
        self.index = index

    def __int__(self):
        return self.index

    def __eq__(self, other):
        return self.name == other.name and self.gender == other.gender

    @classmethod
    def class_length(class_):
        speakers = [v for n, v in vars(class_).items() if isinstance(v, Speaker)]
        return len(speakers)

    def __repr__(self):
        return '%s(name=\'%s\', gender=%s, index=%d)' % (self.__class__.__name__, self.name,
                                                         self.gender.name, self.index)


_speaker_args = {
    'JUDY_BIEBER': ('Judy Bieber', Gender.FEMALE),
    'MARY_ANN': ('Mary Ann', Gender.FEMALE),
    'ELLIOT_MILLER': ('Elliot Miller', Gender.MALE),
    'HILARY_NORIEGA': ('Hilary Noriega', Gender.FEMALE),
    'LINDA_JOHNSON': ('Linda Johnson', Gender.FEMALE)
}
for index, (key, args) in enumerate(_speaker_args.items()):
    setattr(Speaker, key, Speaker(*args, index=index))

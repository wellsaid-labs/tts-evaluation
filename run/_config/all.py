import config as cf

from run._config import audio, data, environment, lang, models


def configure():
    """Configure required modules."""
    cf.enable_fast_trace()
    environment.configure()
    audio.configure()
    models.configure()
    data.configure()
    lang.configure()

import config as cf

from run._config import audio, data, environment, lang, models


def configure(overwrite: bool = False):
    """Configure required modules."""
    cf.enable_fast_trace()
    environment.configure(overwrite=overwrite)
    audio.configure(overwrite=overwrite)
    models.configure(overwrite=overwrite)
    data.configure(overwrite=overwrite)
    lang.configure(overwrite=overwrite)

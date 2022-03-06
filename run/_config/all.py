import sys

import config

from run._config import audio, data, environment, lang, models


def configure():
    """Configure required modules."""
    # TODO: Consider removing after configuration is setup.
    sys.setprofile(config.profile)
    environment.configure()
    audio.configure()
    models.configure()
    data.configure()
    lang.configure()

from run._config import audio, data, environment, lang, models


def configure():
    """Configure required modules."""
    environment.configure()
    audio.configure()
    models.configure()
    data.configure()
    lang.configure()

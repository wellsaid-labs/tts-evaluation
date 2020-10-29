from run.train import spectrogram_model


def test__configure():
    """ Test `spectrogram_model._configure` finds and configures modules. """
    spectrogram_model._configure({})

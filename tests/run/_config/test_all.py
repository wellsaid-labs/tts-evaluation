import config as cf

from run._config.all import configure


def test_configure():
    """Test `run._config.all.configure` finds and configures modules."""
    configure()
    cf.purge()

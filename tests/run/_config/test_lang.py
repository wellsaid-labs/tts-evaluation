import functools

from run._config import is_sound_alike
from run.data._loader import Language


def test_is_sound_alike():
    """Test `is_sound_alike` if determines if two phrase sound-alike."""
    _isa = functools.partial(is_sound_alike, language=Language.ENGLISH)
    assert not _isa("Hello", "Hi")
    assert _isa("financingA", "financing a")
    assert _isa("twentieth", "20th")
    assert _isa("screen introduction", "screen--Introduction,")
    assert _isa("Hello-you've", "Hello. You've")
    assert _isa("'is...'", "is")
    assert _isa("Pre-game", "pregame")
    assert _isa("Dreamfields.", "dream Fields")
    assert _isa(" — ", "")

    # NOTE: These cases are not supported, yet.
    assert not _isa("fifteen", "15")
    assert not _isa("forty", "40")


def test_is_sound_alike__de():
    """Test `is_sound_alike` if determines if two phrase sound-alike in German cases."""
    _isa = functools.partial(is_sound_alike, language=Language.GERMAN)
    assert not _isa("Hänsel", "Gretel")
    assert _isa("Pimpel, Schlafmütz und -- Seppl", "pimpel schlafmütz und seppl")
    assert _isa("Die sieben Zwerge sind Brummbär...", "Die sieben Zwerge sind: Brummbär")
    assert _isa("fließen die Straße", "fliessen die Strasse")

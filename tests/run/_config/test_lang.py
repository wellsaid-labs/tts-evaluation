import functools

from run._config import (
    is_normalized_vo_script,
    is_sound_alike,
    normalize_and_verbalize_text,
    normalize_vo_script,
)
from run._config.lang import _get_long_abbrevs
from run.data._loader import Language


def test_is_sound_alike():
    """Test `is_sound_alike` determines if two phrases sound-alike."""
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
    assert _isa("verite", "vérité")
    assert _isa("fête", "Fete,")

    # NOTE: These cases are not supported, yet.
    assert not _isa("fifteen", "15")
    assert not _isa("forty", "40")


def test_is_sound_alike__de():
    """Test `is_sound_alike` determines if two phrases sound-alike in German cases."""
    _isa = functools.partial(is_sound_alike, language=Language.GERMAN)
    assert not _isa("Hänsel", "Gretel")
    assert _isa("Pimpel, Schlafmütz und -- Seppl", "pimpel schlafmütz und seppl")
    assert _isa("Die sieben Zwerge sind Brummbär...", "Die sieben Zwerge sind: Brummbär")
    assert _isa("fließen die Straße", "fliessen die Strasse")


def test_normalize_vo_script():
    """Test `normalize_vo_script` normalizes scripts appropriately for each language."""
    tests = {
        Language.ENGLISH: (
            "I always refer to my résumé so the hiring manager doesn't assume my responses are "
            "just a façade and can be confident in offering me a 50€/hr wage 🤩.",
            "I always refer to my résumé so the hiring manager doesn't assume my responses are "
            "just a façade and can be confident in offering me a 50EUR/hr wage.",
        ),
        Language.GERMAN: (
            "»…die Straße, Wir gehen am Dienstag.«",
            '"...die Straße, Wir gehen am Dienstag."',
        ),
        Language.SPANISH: (
            "¿Pueden los niños unirse al círculo de juego?",
            "¿Pueden los niños unirse al círculo de juego?",
        ),
        Language.PORTUGUESE: (
            "As crianças podem participar do círculo do jogo?",
            "As crianças podem participar do círculo do jogo?",
        ),
    }

    assert all(
        (text_out == normalize_vo_script(text_in, lang) for text_in, text_out in texts)
        for lang, texts in tests.items()
    )


def test_is_normalized_vo_script():
    """Test `is_normalized_vo_script` handles all languages."""
    tests = {
        Language.ENGLISH: "I always refer to my résumé so the hiring manager doesn't assume my "
        "responses are just a façade and can be confident in offering me a 50€/hr wage 🤩.",
        Language.GERMAN: "»…die Straße, Wir gehen am Dienstag.«",
        Language.SPANISH: "¿Pueden los niños unirse al círculo de juego?",
        Language.PORTUGUESE: "As crianças podem participar do círculo do jogo?",
    }
    assert all(
        is_normalized_vo_script(normalize_vo_script(text, lang), lang)
        for lang, text in tests.items()
    )


def text_normalize_and_verbalize_text():
    """Test `normalize_and_verbalize_text` given the English langauge parameter."""
    text_in, text_out = (
        "7:25AM. Run 13mi, eat 2,000cal, nap for 10-15 minutes, and eat dinner with Dr. "
        "Amelia Fern at 7:00 tonight.",
        "seven twenty-five AM. Run thirteen miles, eat two thousand calories, nap for ten to "
        "fifteen minutes, and eat dinner with Doctor Amelia Fern at seven oh clock tonight.",
    )

    assert text_out == normalize_and_verbalize_text(text_in, Language.ENGLISH)


def test__get_long_abbrevs():
    """Test that `_get_long_abbrevs` is able to get abbreviations that take a long time to speak."""
    tests = (
        ("ABC", ["ABC"]),
        ("NDAs", ["NDA"]),
        ("HAND-CUT", ["HAND-CUT"]),
        ("NOVA/national", ["NOVA"]),
        ("I V As?", ["I V A"]),
        ("I.V.A.", ["I.V.A."]),
        ("information...ELEVEN", ["ELEVEN"]),
        ("(JNA)", ["JNA"]),
        ("JN a", ["JN"]),
        ("PwC", ["PwC"]),
        ("JC PENNEY", ["JC PENNEY"]),
        ("JCPenney", ["JC"]),
        ("DirecTV", ["TV"]),
        ("M*A*C", ["M*A*C"]),
        ("fMRI", ["MRI"]),
        ("RuBP.", ["RuBP"]),
        ("MiniUSA.com,", ["USA"]),
        ("7-Up", []),
        ("7UP", ["7UP"]),
        ("NDT", ["NDT"]),
        ("ND T", ["ND T"]),
        ("L.V.N,", ["L.V.N"]),
        ("I", []),
        ("p.m.", ["p.m."]),
        ("place...where", []),
        ("Smucker's.", []),
        ("DVD-Players", ["DVD"]),
        ("PCI-DSS,", ["PCI-DSS"]),
        ("UFO's,", ["UFO"]),
        ("most[JT5]", ["JT5"]),
        ("NJ--at", ["NJ"]),
        ("U. S.", ["U. S"]),
        ("ADHD.Some", ["ADHD"]),
        ("W-USA", ["W-USA"]),
        ("P-S-E-C-U", ["P-S-E-C-U"]),
        ("J. V.", ["J. V"]),
        ("P.m.", ["P.m."]),
        ("Big-C", []),
        ("Big C.", []),
        ("U-Boats", []),
        ("well.I'll,", []),
        ("well. I'll", []),
        ("Rain-x-car", []),
        ("Rain-X car", []),
        ("L.L.Bean", ["L.L"]),
        ("WBGP -", ["WBGP"]),
        ("W BG P.", ["W BG P"]),
        ("K RC K", ["K RC K"]),
        ("DVD-L10", ["DVD-L10"]),
        ("DVD L10", ["DVD L10"]),
        ("DVD-L10", ["DVD-L10"]),
        ("DVD, L10", ["DVD", "L10"]),
        ("t-shirt", []),
        ("T-shirt", []),
    )

    for in_, out in tests:
        assert list(_get_long_abbrevs(in_)) == out

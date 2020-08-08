from src.bin.sync_script_with_audio import _get_speech_context_phrases


def test___get_speech_context_phrases():
    assert set(_get_speech_context_phrases(['a b c d e f g h i j'],
                                           5).phrases) == set(['a b c', 'd e f', 'g h i', 'j'])

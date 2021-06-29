import lib


def maybe_normalize_audio_and_cache_side_effect(metadata: lib.audio.AudioMetadata):
    """`run.data._loader.normalize_audio_and_cache.side_effect` that returns the path without
    normalization."""
    return metadata.path

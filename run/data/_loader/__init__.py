from run.data._loader import english, german, m_ailabs, portuguese, spanish, structures, utils
from run.data._loader.structures import (
    Alignment,
    Language,
    NonalignmentSpans,
    Passage,
    Session,
    Span,
    Speaker,
    alignment_dtype,
    has_a_mistranscription,
    voiced_nonalignment_spans,
)
from run.data._loader.utils import (
    DataLoader,
    DataLoaders,
    SpanGenerator,
    conventional_dataset_loader,
    dataset_loader,
    get_non_speech_segments_and_cache,
    is_normalized_audio_file,
    maybe_normalize_audio_and_cache,
    normalize_audio,
    normalize_audio_suffix,
    read_audio,
)

DATASETS: DataLoaders = {
    **portuguese.DATASETS,
    **spanish.DATASETS,
    **german.DATASETS,
    **english.DATASETS,
}
WSL_DATASETS: DataLoaders = {
    **portuguese.WSL_DATASETS,
    **spanish.WSL_DATASETS,
    **german.WSL_DATASETS,
    **english.WSL_DATASETS,
}
LIBRIVOX_DATASETS: DataLoaders = {
    **portuguese.LIBRIVOX_DATASETS,
    **spanish.M_AILABS_DATASETS,
    **german.M_AILABS_DATASETS,
    **english.M_AILABS_DATASETS,
}
LIBRIVOX_DATASETS[english.lj_speech.LINDA_JOHNSON] = english.lj_speech.lj_speech_dataset
RND_DATASETS: DataLoaders = {**english.RND_DATASETS}
DICTIONARY_DATASETS: DataLoaders = {
    english.dictionary.GCP_SPEAKER: english.dictionary.dictionary_dataset
}

__all__ = [
    "english",
    "german",
    "m_ailabs",
    "portuguese",
    "spanish",
    "structures",
    "utils",
    "Alignment",
    "Language",
    "NonalignmentSpans",
    "Passage",
    "Session",
    "Span",
    "Speaker",
    "alignment_dtype",
    "has_a_mistranscription",
    "voiced_nonalignment_spans",
    "DataLoader",
    "DataLoaders",
    "SpanGenerator",
    "conventional_dataset_loader",
    "dataset_loader",
    "get_non_speech_segments_and_cache",
    "is_normalized_audio_file",
    "maybe_normalize_audio_and_cache",
    "normalize_audio",
    "normalize_audio_suffix",
    "read_audio",
    "DATASETS",
    "DICTIONARY_DATASETS",
    "LIBRIVOX_DATASETS",
    "RND_DATASETS",
    "WSL_DATASETS",
]

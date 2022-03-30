""" Script for generating an alignment between a script and a voice-over.

Given a voice-over and a voice-over script, this python script generates an alignment between text
spans and audio spans. Also this script will help detect inconsistencies between the two.

TODO: Support creating a dataset with just a voice-over using Google STT for the transcript.

Usage Example:

    PREFIX=gs://wellsaid_labs_datasets/hilary_noriega/processed
    python -m run.data.sync_script_with_audio \
      --voice-over "$PREFIX/recordings/script_1.wav" \
      --script "$PREFIX/scripts/script_1_-_hilary.csv" \
      --destination "$PREFIX/"

Usage Example (German):
    PREFIX=gs://wellsaid_labs_datasets/experiment__german__unser_weld
    python -m run.data.sync_script_with_audio \
      --voice-over "$PREFIX/processed/speech_to_text/*.wav" \
      --script "$PREFIX/processed/scripts/*.csv" \
      --destination "$PREFIX/processed_local/" \
      --language German
"""
import dataclasses
import json
import logging
import re
import time
import typing
from copy import deepcopy
from functools import lru_cache, partial
from io import StringIO
from itertools import chain, groupby, repeat
from typing import cast

import pandas
import tabulate
import typer
from google.cloud import speech_v1p1beta1 as speech
from google.cloud import storage
from google.cloud.speech_v1p1beta1 import (
    LongRunningRecognizeResponse,
    RecognitionAudio,
    SpeechContext,
)
from google.protobuf.json_format import MessageToDict
from Levenshtein import distance  # type: ignore
from tqdm import tqdm

import lib
import run
from lib.environment import AnsiCodes
from lib.utils import flatten_2d, get_chunks
from run import _config
from run._utils import blob_to_gcs_uri, gcs_uri_to_blob
from run.data._loader import Language

lib.environment.set_basic_logging_config()
logger = logging.getLogger(__name__)


class ScriptToken(typing.NamedTuple):
    """
    Args:
        script_index: The index of the script from which the text comes from.
        text: The script text.
        slice: The start and end script character index of `text`.
    """

    script_index: int
    text: typing.Optional[str] = None
    slice: typing.Tuple[typing.Optional[int], typing.Optional[int]] = (None, None)


class SttToken(typing.NamedTuple):
    """
    Args:
        text: The predicted STT text.
        audio: The start and end seconds of the corresponding audio snippet.
        slice: The start and end transcript character index of `text`.
    """

    text: str
    audio: typing.Tuple[float, float]
    slice: typing.Tuple[int, int]


# TODO: Use `pydantic` for loading speech-to-text results into a data structure, learn more:
# https://pydantic-docs.helpmanual.io/. It'll also validate the data during runtime.


class _SttWord(typing.TypedDict):
    startTime: str
    endTime: str
    word: str


class _SttAlternative(typing.TypedDict):
    transcript: str
    confidence: float
    words: typing.List[_SttWord]


class _SttAlternatives(typing.TypedDict):
    alternatives: typing.List[_SttAlternative]
    languageCode: _config.LanguageCode


class SttResult(typing.TypedDict):
    """The expected typing of a Google Speech-to-Text call."""

    results: typing.List[_SttAlternatives]


class Alignments(typing.TypedDict):
    """
    Args:
        transcript: The audio transcript used to align the script to the voice-over.
        alignments: For each script, this contains a list of alignments.
    """

    transcript: str
    alignments: typing.List[typing.List[run.data._loader.Alignment]]


@dataclasses.dataclass
class Stats:
    """
    Args:
        total_aligned_tokens: The total number of tokens aligned.
        total_tokens: The total number of tokens processed.
        sound_alike: Set of all tokens that sound-a-like.
        ...
    """

    total_aligned_tokens: int = 0
    total_tokens: int = 0
    sound_alike: typing.Set[frozenset] = dataclasses.field(default_factory=set)
    script_unaligned: typing.Set[str] = dataclasses.field(default_factory=set)
    transcript_unaligned: typing.Set[str] = dataclasses.field(default_factory=set)

    def log(self):
        total_unaligned_tokens = self.total_tokens - self.total_aligned_tokens
        words_unaligned = format_ratio(total_unaligned_tokens, self.total_tokens)
        logger.info(f"Total word(s) unaligned: {words_unaligned}")

        sound_alike = [tuple(p) for p in self.sound_alike]
        sound_alike = [(distance(*p), *p) for p in sound_alike]
        headers = ["Edit Distance", "", ""]
        sound_alike_ = tabulate.tabulate(sorted(sound_alike, reverse=True)[:50], headers=headers)
        logger.info(f"Most different sound-a-like word(s):\n{sound_alike_}")

        _quote = f'{AnsiCodes.DARK_GRAY}"{AnsiCodes.RESET_ALL}'
        quote = lambda s: f"{_quote}{s}{_quote}"
        script_unaligned = [quote(s.strip()) for s in self.script_unaligned]
        logger.info(
            f"Longest {AnsiCodes.RED}unaligned script span(s){AnsiCodes.RESET_ALL}:\n%s",
            "\n".join(sorted(script_unaligned, key=len, reverse=True)[:50]),
        )
        transcript_unaligned = [quote(s.strip()) for s in self.transcript_unaligned]
        logger.info(
            f"Longest {AnsiCodes.GREEN}unaligned transcript span(s){AnsiCodes.RESET_ALL}:\n%s",
            "\n".join(sorted(transcript_unaligned, key=len, reverse=True)[:50]),
        )


STATS = Stats()
CONTROL_CHARACTERS_REGEX = re.compile(r"[\x00-\x08\x0b\x0c\x0d\x0e-\x1f]")

normalize_vo_script = lru_cache(maxsize=2 ** 20)(_config.normalize_vo_script)


def format_ratio(a: float, b: float) -> str:
    """
    Example:
        >>> format_ratio(1, 100)
        '1.000000% [1 of 100]'
    """
    return f"{(float(a) / b) * 100}% [{a} of {b}]"


def _minus(text: str):
    """Helper function for `_format_gap`."""
    STATS.script_unaligned.add(text)
    return f'\n{AnsiCodes.RED}--- "{text}"{AnsiCodes.RESET_ALL}\n'


def _plus(text: str):
    """Helper function for `_format_gap`."""
    STATS.transcript_unaligned.add(text)
    return f'{AnsiCodes.GREEN}+++ "{text}"{AnsiCodes.RESET_ALL}\n'


def _format_gap(
    scripts: typing.List[str], tokens: typing.List[ScriptToken], stt_tokens: typing.List[SttToken]
) -> typing.Iterable[str]:
    """Format the span of `tokens` and `stt_tokens` that lie between two alignments, the gap."""
    if len(tokens) > 2 or len(stt_tokens) > 0:
        if tokens[0].script_index != tokens[-1].script_index:
            yield _minus(scripts[tokens[0].script_index][tokens[0].slice[-1] :])
            yield "=" * 100
            for i in range(tokens[0].script_index + 1, tokens[-1].script_index):
                yield _minus(scripts[i])
                yield "=" * 100
            yield _minus(scripts[tokens[-1].script_index][: tokens[-1].slice[0]])
        else:
            yield _minus(scripts[tokens[0].script_index][tokens[0].slice[-1] : tokens[-1].slice[0]])
        text = " ".join([t.text for t in stt_tokens])
        yield _plus(text)
    elif len(tokens) == 2 and tokens[0].script_index == tokens[-1].script_index:
        yield scripts[tokens[0].script_index][tokens[0].slice[-1] : tokens[-1].slice[0]]
    elif len(tokens) == 2:
        yield scripts[tokens[0].script_index][tokens[0].slice[-1] :]
        yield "\n" + "=" * 100 + "\n"
        yield scripts[tokens[-1].script_index][: tokens[-1].slice[0]]


def format_differences(
    scripts: typing.List[str],
    alignments: typing.List[typing.Tuple[int, int]],
    tokens: typing.List[ScriptToken],
    stt_tokens: typing.List[SttToken],
) -> typing.Iterable[str]:
    """Format the differences between `tokens` and `stt_tokens` given the `alignments`.

    Args:
        scripts: The voice-over scripts.
        alignments: Alignments between `tokens` and `stt_tokens`.
        tokens: Tokens derived from `scripts`.
        stt_tokens: The predicted speech-to-text tokens.

    Returns: Generates `str` snippets that can be joined together for the full output.

    Example Output:
    > Welcome to Morton Arboretum, home to more than
    > --- "36 HUNDRED "
    > +++ "3,600"
    > native trees, shrubs, and plants.
    """
    tokens = [ScriptToken(slice=(None, 0), script_index=0)] + tokens
    tokens += [ScriptToken(slice=(len(scripts[-1]), None), script_index=len(scripts) - 1)]
    alignments = [(a + 1, b) for a, b in alignments]

    groups: typing.List[typing.Tuple[typing.Optional[int], typing.List[typing.Tuple[int, int]]]]
    groups = [(None, [(0, -1)])]
    groups += [(i, list(g)) for i, g in groupby(alignments, lambda a: tokens[a[0]].script_index)]
    groups += [(None, [(len(tokens) - 1, len(stt_tokens))])]

    current = groups[0][1]
    for (_, before), (i, current), (_, after) in zip(groups, groups[1:], groups[2:]):
        i = cast(int, i)
        is_unaligned = zip([before[-1]] + current, current + [after[0]])
        if any([b[0] - a[0] > 1 or b[1] - a[1] > 1 for a, b in is_unaligned]):
            yield from _format_gap(
                scripts,
                tokens[before[-1][0] : current[0][0] + 1],
                stt_tokens[before[-1][1] + 1 : current[0][1]],
            )
            for last, next_ in zip(current, current[1:] + [None]):  # type: ignore
                yield scripts[i][tokens[last[0]].slice[0] : tokens[last[0]].slice[-1]]
                if next_:
                    yield from _format_gap(
                        scripts,
                        tokens[last[0] : next_[0] + 1],
                        stt_tokens[last[1] + 1 : next_[1]],
                    )

    yield from _format_gap(scripts, tokens[current[-1][0] :], stt_tokens[current[-1][1] + 1 :])


def _fix_alignments(
    scripts: typing.List[str],
    alignments: typing.List[typing.Tuple[int, int]],
    tokens: typing.List[ScriptToken],
    stt_tokens: typing.List[SttToken],
    language: Language,
) -> typing.Tuple[
    typing.List[ScriptToken], typing.List[SttToken], typing.List[typing.Tuple[int, int]]
]:
    """Resolve misalignments if unaligned `tokens` and unaligned `stt_tokens` sound alike.

    Example: This will align these two tokens:
      `ScriptToken("Hello.You've")`
      `SttToken("Hello You\'ve"))`
    """
    stt_tokens = stt_tokens.copy()
    alignments = alignments.copy()
    iterator = zip([(int(-1), int(-1))] + alignments, alignments + [(len(tokens), len(stt_tokens))])
    for i, (last, next_) in reversed(list(enumerate(iterator))):
        has_gap = next_[0] - last[0] > 1 and next_[1] - last[1] > 1
        if not (has_gap and tokens[last[0] + 1].script_index == tokens[next_[0] - 1].script_index):
            continue

        script_index = tokens[last[0] + 1].script_index
        slice_ = (tokens[last[0] + 1].slice[0], tokens[next_[0] - 1].slice[-1])
        token = ScriptToken(script_index, scripts[script_index][slice(*slice_)], slice_)
        stt_token = SttToken(
            " ".join([t.text for t in stt_tokens[last[1] + 1 : next_[1]]]),
            (stt_tokens[last[1] + 1].audio[0], stt_tokens[next_[1] - 1].audio[-1]),
            (stt_tokens[last[1] + 1].slice[0], stt_tokens[next_[1] - 1].slice[-1]),
        )

        assert token.text is not None
        if _config.is_sound_alike(token.text, stt_token.text, language):
            logger.info(f'Fixing alignment between: "{token.text}" and "{stt_token.text}"')
            for j in reversed(range(last[0] + 1, next_[0])):
                del tokens[j]
                alignments = [(a - 1, b) if a > j else (a, b) for a, b in alignments]
            tokens.insert(last[0] + 1, token)
            alignments = [(a + 1, b) if a >= last[0] + 1 else (a, b) for a, b in alignments]

            for j in reversed(range(last[1] + 1, next_[1])):
                del stt_tokens[j]
                alignments = [(a, b - 1) if b > j else (a, b) for a, b in alignments]
            stt_tokens.insert(last[1] + 1, stt_token)
            alignments = [(a, b + 1) if b >= last[1] + 1 else (a, b) for a, b in alignments]

            alignments.insert(i, (last[0] + 1, last[1] + 1))

    return tokens, stt_tokens, alignments


_default_get_window_size: typing.Callable[[int, int], int]
_default_get_window_size = lambda a, b: int(max(round(a * 0.05), 256) + abs(a - b))


def _flatten_stt_result(stt_result: SttResult) -> typing.Tuple[str, typing.List[SttToken]]:
    """Flatten a `SttResult` into a list of `SttToken` tokens and a transcript."""
    stt_tokens = []
    transcript = ""
    offset = 0
    for result in [r["alternatives"][0] for r in stt_result["results"] if r["alternatives"][0]]:
        transcript += result["transcript"]
        for word in result["words"]:
            audio = (float(word["startTime"][:-1]), float(word["endTime"][:-1]))
            sliced = transcript[slice(offset, offset + len(word["word"]))]
            if sliced != word["word"]:
                message = "The transcript ('%s') doesn't align with the words ('%s')."
                logging.warning(message, sliced, word["word"])
                offset -= 1  # NOTE: Google may have forgotten a space, so we corrected for that.
            stt_tokens.append(SttToken(word["word"], audio, (offset, offset + len(word["word"]))))
            offset += len(word["word"]) + 1
    for prev, next in zip(stt_tokens[:-1], stt_tokens[1:]):
        if prev.audio[-1] > next.audio[0]:
            # NOTE: Unfortunately, in rare cases, this does happen.
            logging.warning("These alignments overlap: %s > %s", prev, next)
    # NOTE: Unfortunately, Google STT, in rare cases, will predict a token with punctuation, for
    # example:
    # "Military theory and practice contributed approaches to managing the newly-popular
    # factories.given."
    # Google STT predicted the token "factories.given." in the above transcript it predicted.
    if " ".join(t.text for t in stt_tokens).strip() != transcript.strip():
        logger.warning("Google STT may not be white-space tokenized.")
    message = "Transcript has white-spaces."
    no_white_space = lambda t: t.strip() == t
    assert all(no_white_space(transcript[t.slice[0] : t.slice[1]]) for t in stt_tokens), message
    return transcript, stt_tokens


def align_stt_with_script(
    scripts: typing.List[str],
    stt_result: SttResult,
    language: Language,
    get_window_size: typing.Callable[[int, int], int] = _default_get_window_size,
) -> Alignments:
    """Align an STT result(s) with the related script(s). Uses white-space tokenization.

    NOTE: The `get_window_size` default used with `align_tokens` was set based off empirical
    data. The three components are:
      - `a * 0.05` assumes that a maximum of 5% of tokens will be misaligned.
      - `max(x, 256)` is a minimum window size that doesn't cause performance issues.
      - `abs(a - b)` is an additional bias for large misalignments.

    Args:
        scripts: The voice-over script.
        stt_result: The speech-to-text results for the related voice-over.
        get_window_size: Callable for computing the maximum window size in tokens for aligning the
            STT results with the provided script.
    """
    script_tokens_ = [
        [ScriptToken(i, m.group(0), (m.start(), m.end())) for m in re.finditer(r"\S+", script)]
        for i, script in enumerate(scripts)
    ]
    script_tokens: typing.List[ScriptToken] = flatten_2d(script_tokens_)
    transcript, stt_tokens = _flatten_stt_result(stt_result)

    # Align `script_tokens` and `stt_tokens`.
    args = (
        [normalize_vo_script(cast(str, t.text).lower(), language) for t in script_tokens],
        [normalize_vo_script(t.text.lower(), language) for t in stt_tokens],
        get_window_size(len(script_tokens), len(stt_tokens)),
    )
    is_sound_alike = partial(_config.is_sound_alike, language=language)
    alignments = lib.text.align_tokens(*args, allow_substitution=is_sound_alike)[1]
    # TODO: Should `_fix_alignments` align between scripts? Is that data valuable?
    script_tokens, stt_tokens, alignments = _fix_alignments(
        scripts, alignments, script_tokens, stt_tokens, language
    )

    # Log statistics
    STATS.total_aligned_tokens += len(alignments)
    STATS.total_tokens += len(script_tokens)
    logger.info(
        "Script word(s) unaligned: %s",
        format_ratio(len(script_tokens) - len(alignments), len(script_tokens)),
    )
    logger.info(
        "Failed to align:\n%s",
        "".join(format_differences(scripts, alignments, script_tokens, stt_tokens)),
    )

    return_ = Alignments(transcript=transcript, alignments=[[] for _ in range(len(scripts))])
    for alignment in alignments:
        stt_token = stt_tokens[alignment[1]]
        script_token = script_tokens[alignment[0]]
        alignment_ = run.data._loader.Alignment(
            script=cast(typing.Tuple[int, int], script_token.slice),
            audio=stt_token.audio,
            transcript=stt_token.slice,
        )
        return_["alignments"][script_token.script_index].append(alignment_)
    return return_


def _get_speech_context(
    script: str,
    language: Language,
    max_phrase_length: int = 100,
    max_overlap: float = 0.25,
) -> SpeechContext:
    """Given the voice-over script generate `SpeechContext` to help Speech-to-Text recognize
    specific words or phrases more frequently.

    NOTE:
    - Google STT as of June 21st, 2019 tends to tokenize based on white spaces.
    - In August 2020, `boost` reduced the accuracy of STT even with values as high was 1000, much
      larger than the recommended value of 20.
    - The maximum phrase length Google STT accepts is 100 characters.
    - Adding overlap between phrases helps perserve script continuity. There are no guidelines on
      how much overlap is best.

    Args:
        script
        max_phrase_length: The maximum character length of a phrase.
        max_overlap: The maximum overlap between two consecutive phrases.

    Example:
        >>> sorted(_get_speech_context('a b c d e f g h i j', 5, 0.0).phrases)
        ['a b c', 'd e f', 'g h i', 'j']
        >>> sorted(_get_speech_context('a b c d e f g h i j', 5, 0.2).phrases)
        ['a b c', 'c d e', 'e f g', 'g h i', 'i j']

    """
    iter_ = re.finditer(r"\S+", normalize_vo_script(script, language))
    spans = [(m.start(), m.end()) for m in iter_]
    phrases = []
    start, next_start = 0, 0
    for prev, next in zip(spans, spans[1:]):
        if next_start == start and next[0] - start >= round(max_phrase_length * (1 - max_overlap)):
            next_start = next[0]
        if next[1] - start > max_phrase_length:
            if prev[1] - start <= max_phrase_length:
                phrases.append(script[start : prev[1]])
            start = next_start
    phrases.append(script[start:])
    return SpeechContext(phrases=list(set(phrases)))


def _run_stt(
    audio_blobs: typing.List[storage.Blob],
    scripts: typing.List[str],
    dest_blobs: typing.List[storage.Blob],
    poll_interval: float,
    language: Language,
    add_speech_context: bool = False,
):
    """Helper function for `run_stt`.

    NOTE: Speech context helps speech recognition, a lot, and it decreases timestamp accuracy,
    a lot. See this issue: https://issuetracker.google.com/u/1/issues/174239874
    TODO: Speech context might be important for verbalized numbers. Google will return
    numbers in a standard format which will not align with the verbalized format. For example,
    these didn't align:
    " three hundred and seventy-five ", "375"
    " one hundred dollars ", "$100"
    " one hundred percent ", "100%"
    """
    stt_config = _config.STT_CONFIGS[language]
    operations = []
    for audio_blob, script, dest_blob in zip(audio_blobs, scripts, dest_blobs):
        config = deepcopy(stt_config)
        if add_speech_context:
            speech_context = _get_speech_context("\n".join(script), language)
            config.speech_contexts.append(speech_context)  # type: ignore
        audio = RecognitionAudio(uri=blob_to_gcs_uri(audio_blob))
        operations.append(speech.SpeechClient().long_running_recognize(config=config, audio=audio))
        logger.info(f"STT running in {stt_config.language_code}, {stt_config.model} model...")
        message = f"STT operation {operations[-1].operation.name}"
        message += f'"{blob_to_gcs_uri(dest_blob)}" started.'
        logger.info(message)

    progress = [0] * len(operations)
    progress_bar = tqdm(total=100)
    while not all(o is None for o in operations):
        for i, operation in enumerate(operations):
            if operation is None:
                continue

            metadata = operation.metadata
            if operation.done():
                # Learn more:
                # https://stackoverflow.com/questions/64470470/how-to-convert-google-cloud-natural-language-entity-sentiment-response-to-json-d
                response: LongRunningRecognizeResponse = operation.result()
                json_ = MessageToDict(LongRunningRecognizeResponse.pb(response))
                # TODO: Along with `upload_from_string`, add blob metadata with the creation
                # time. Google Storage has a strange implementation of creation date, and it'd be
                # useful to add our own creation date timestamp, that doesn't get overwritten.
                dest_blobs[i].upload_from_string(json.dumps(json_), content_type="application/json")
                message = f"STT operation {operation.operation.name} "
                message += f'"{blob_to_gcs_uri(dest_blobs[i])}" finished.'
                logger.info(message)
                operations[i] = None
                progress[i] = 100
            elif metadata is not None and metadata.progress_percent is not None:
                progress[i] = metadata.progress_percent
            progress_bar.update(min(progress) - progress_bar.n)
            time.sleep(poll_interval)


def run_stt(
    audio_blobs: typing.List[storage.Blob],
    scripts: typing.List[str],
    dest_blobs: typing.List[storage.Blob],
    language: Language,
    poll_interval: float = 1 / 10,
    max_connections: int = 256,
):
    """Run speech-to-text on `audio_blobs` and save them at `dest_blobs`.

    NOTE:
    - The documentation for `RecognitionConfig` does not include all the available
      options. All the potential parameters are included here:
      https://cloud.google.com/speech-to-text/docs/reference/rest/v1p1beta1/RecognitionConfig
    - Time offset values are only included for the first alternative provided in the recognition
      response, learn more here: https://cloud.google.com/speech-to-text/docs/async-time-offsets
    - Careful not to exceed default 300 requests a second quota. If you do exceed it, note that
      Google STT will not respond to requests for up to 30 minutes.
    - The `stt_config` defaults are set based on:
      https://blog.timbunce.org/2018/05/15/a-comparison-of-automatic-speech-recognition-asr-systems/
    - You can preview the running operations via the command line. For example:
      $ gcloud ml speech operations describe 5187645621111131232
    - Google STT with `enable_automatic_punctuation=False` still predicts hyphens. This can cause
      issues downstream if a non-hyphenated word is expected.
    - The progress bar tracks the progress of the slowest operation since all the operations
      run in parallel.

    TODO: Look into how many requests are made every poll, in order to better calibrate against
    Google's quota.

    Args:
        audio_blobs: List of GCS voice-over blobs.
        scripts: List of voice-over scripts. These are used to give hints to the STT.
        dest_blobs: List of GCS blobs to upload results too.
        poll_interval: The interval between each poll of STT progress.
        stt_config
        max_connections: The maximum number of connections to create at a time, learn more:
            https://unix.stackexchange.com/questions/36841/why-is-number-of-open-files-limited-in-linux
            https://unix.stackexchange.com/questions/157351/why-are-tcp-ip-sockets-considered-open-files
    """
    assert len(audio_blobs) == len(scripts)
    assert len(audio_blobs) == len(dest_blobs)
    chunk = partial(get_chunks, n=max_connections)
    batches = list(zip(chunk(audio_blobs), chunk(scripts), chunk(dest_blobs)))
    logger.info(f"Running {len(audio_blobs)} STT operation(s) in {len(batches)} batch(es).")
    for args in batches:
        _run_stt(*args, poll_interval, language)


def _sync_and_upload(
    audio_blobs: typing.List[storage.Blob],
    script_blobs: typing.List[storage.Blob],
    dest_blobs: typing.List[storage.Blob],
    stt_blobs: typing.List[storage.Blob],
    alignment_blobs: typing.List[storage.Blob],
    text_column: str,
    recorder: lib.environment.RecordStandardStreams,
    language: Language,
):
    """Sync `script_blobs` with `audio_blobs` and upload to `alignment_blobs`."""
    assert len(audio_blobs) == len(script_blobs), "Expected the same number of blobs."
    assert len(audio_blobs) == len(dest_blobs), "Expected the same number of blobs."
    assert len(audio_blobs) == len(stt_blobs), "Expected the same number of blobs."
    assert len(audio_blobs) == len(alignment_blobs), "Expected the same number of blobs."

    logger.info("Downloading voice-over scripts...")
    scripts_ = [s.download_as_bytes().decode("utf-8") for s in script_blobs]
    scripts: typing.List[typing.List[str]] = [
        cast(pandas.DataFrame, pandas.read_csv(StringIO(s)))[text_column].tolist() for s in scripts_
    ]
    message = "Some or all script(s) are missing or incorrecly formatted."
    assert all(isinstance(t, str) for t in flatten_2d(scripts)), message
    message = "Script(s) cannot contain funky characters."
    assert all(_config.is_normalized_vo_script(t, language) for t in flatten_2d(scripts)), message

    logger.info("Maybe running speech-to-text and caching results...")
    filtered = list(filter(lambda i: not i[-1].exists(), zip(audio_blobs, scripts, stt_blobs)))
    if len(filtered) > 0:
        args = cast(
            typing.Tuple[typing.List[storage.Blob], typing.List[str], typing.List[storage.Blob]],
            zip(*filtered),
        )
        run_stt(*args, language)
    stt_results: typing.List[SttResult]
    stt_results = [json.loads(b.download_as_bytes()) for b in stt_blobs]

    for script, stt_result, alignment_blob in zip(scripts, stt_results, alignment_blobs):
        message = f'Running alignment "{blob_to_gcs_uri(alignment_blob)}" and uploading results...'
        logger.info(message)
        alignment = align_stt_with_script(script, stt_result, language)
        alignment_blob.upload_from_string(json.dumps(alignment), content_type="application/json")

    STATS.log()

    # NOTE: `storage.Blob` doesn't implement `__hash__`.
    for dest_gcs_uri in set(blob_to_gcs_uri(b) for b in dest_blobs):
        dest_blob = gcs_uri_to_blob(dest_gcs_uri)
        blob = dest_blob.bucket.blob(cast(str, dest_blob.name) + recorder.log_path.name)
        logger.info(f"Uploading logs to `{blob}`...")
        # Formerly uploaded from string, but `recorder.log_path.read_text()` fails to read unencoded
        # log file with unicode characters. [See TODO in `recorder` instantiation]
        blob.upload_from_filename(recorder.log_path.absolute())


# TODO: Allow for dialects to be selected as well, instead of just `language`.


def main(
    voice_over: typing.List[str] = typer.Option(
        ...,
        help=(
            "GCS link(s) to audio file(s) with a supported audio "
            "encoding (https://cloud.google.com/speech-to-text/docs/encoding)."
        ),
    ),
    script: typing.List[str] = typer.Option(..., help="GCS link(s) to the script(s) CSV files."),
    destination: typing.List[str] = typer.Option(
        ..., help="GCS location(s) to upload alignment(s) and speech-to-text result(s)."
    ),
    text_column: str = typer.Option("Content", help="Column name with script text in --scripts."),
    stt_folder: str = typer.Option(
        "speech_to_text/",
        help="Upload speech-to-text results to this folder in --destinations.",
    ),
    alignments_folder: str = typer.Option(
        "alignments/", help="Upload alignment results to this folder in --destinations."
    ),
    language: Language = typer.Option(
        Language.ENGLISH, help="Specify the language of the dataset being synced."
    ),
):
    """Align --scripts with --voice-overs and upload alignments to --destinations."""
    if len(voice_over) > len(destination):
        ratio = len(voice_over) // len(destination)
        destination = list(chain.from_iterable(repeat(x, ratio) for x in destination))

    message = "There should be the same number of voice-overs (%d) and scripts (%d)."
    assert len(voice_over) == len(script), message % (len(voice_over), len(script))

    # NOTE: Save a log of the execution for future reference
    # TODO: Force encoding to be utf-8 on the system, example:
    # https://stackoverflow.com/questions/1473577/writing-unicode-strings-via-sys-stdout-in-python
    recorder = lib.environment.RecordStandardStreams()

    dest_blobs = [gcs_uri_to_blob(d) for d in destination]
    audio_blobs = [gcs_uri_to_blob(v) for v in voice_over]
    script_blobs = [gcs_uri_to_blob(s) for s in script]
    for item in zip(audio_blobs, script_blobs, dest_blobs):
        args = tuple([blob_to_gcs_uri(blob) for blob in item])
        logger.info('Processing... \n "%s" \n "%s" \n and saving to... "%s"', *args)

    filenames = [cast(str, b.name).split("/")[-1].split(".")[0] + ".json" for b in audio_blobs]
    iter_ = list(zip(dest_blobs, filenames))
    stt_blobs = [b.bucket.blob(cast(str, b.name) + stt_folder + n) for b, n in iter_]
    alignment_blobs = [b.bucket.blob(cast(str, b.name) + alignments_folder + n) for b, n in iter_]

    blobs = (audio_blobs, script_blobs, dest_blobs, stt_blobs, alignment_blobs)
    _sync_and_upload(*blobs, text_column, recorder, language)


if __name__ == "__main__":  # pragma: no cover
    typer.run(main)

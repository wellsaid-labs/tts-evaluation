from __future__ import annotations

from functools import lru_cache
from functools import partial

import logging
import os
import re
import shlex
import subprocess
import typing

from third_party import LazyLoader
from tqdm import tqdm
normalise = LazyLoader('normalise', globals(), 'normalise')

import ftfy
import unidecode
if typing.TYPE_CHECKING:  # pragma: no cover
    import spacy.lang.en
spacy_en = LazyLoader('spacy_en', globals(), 'spacy.lang.en')
en_core_web_md = LazyLoader('en_core_web_md', globals(), 'en_core_web_md')
nltk = LazyLoader('nltk', globals(), 'nltk')
Levenshtein = LazyLoader('Levenshtein', globals(), 'Levenshtein')

import lib

logger = logging.getLogger(__name__)


def _grapheme_to_phoneme_helper(grapheme: str,
                                service: str = 'espeak',
                                flags: typing.List[str] = ['--ipa=3', '-q', '-ven-us', '--stdin'],
                                separator: str = '',
                                service_separator: str = '_') -> str:
    """
    TODO: Since eSpeak does not preserve punctuation or white spaces, we shouldn't preserve
    white spaces via `strip` on the edges.

    Args:
        grapheme
        service: The service used to compute phonemes.
        flags: The list of flags to add to the service.
        separator: The separator used to separate phonemes.
        service_separator: The separator used by the service between phonemes.
    """
    # NOTE: `espeak` can be inconsistent in it's handling of outer spacing; therefore, it's
    # recommended both the `espeak` output and input is trimmed.
    grapheme, stripped_left, stripped_right = strip(grapheme)

    # NOTE: The `--sep` flag is not supported by older versions of `espeak`.
    # NOTE: We recommend using `--stdin` otherwise `espeak` might misinterpret an input like
    # "--For this community," as a flag.
    command = 'echo %s | %s %s' % (shlex.quote(grapheme), service, ' '.join(flags))
    phoneme = subprocess.check_output(command, shell=True).decode('utf-8')
    assert not separator or separator == service_separator or separator not in phoneme, (
        'The separator is not unique.')

    phoneme = ' '.join([s.strip() for s in phoneme.strip().split('\n')])

    # NOTE: Remove language flags like `(en-us)` or `(fr)` that might be included for text like:
    # Grapheme: “MON DIEU”
    # Phoneme: “m_ˈɑː_n (fr)_d_j_ˈø_(en-us)”
    phoneme = re.sub(r'\(.+?\)', '', phoneme)

    # NOTE: Replace multiple separators in a row without any phonemes in between with one separator.
    phoneme = re.sub(r'%s+' % re.escape(service_separator), service_separator, phoneme)
    phoneme = re.sub(r'%s+\s+' % re.escape(service_separator), ' ', phoneme)
    phoneme = re.sub(r'\s+%s+' % re.escape(service_separator), ' ', phoneme)
    phoneme = phoneme.strip()

    phoneme = stripped_left + phoneme + stripped_right
    phoneme = phoneme.replace(service_separator, separator)

    # NOTE: Add separators around stress tokens and words.
    phoneme = phoneme.replace(' ', separator + ' ' + separator)
    phoneme = phoneme.replace('ˈ', separator + 'ˈ' + separator)
    phoneme = phoneme.replace('ˌ', separator + 'ˌ' + separator)
    phoneme = re.sub(r'%s+' %
                     re.escape(separator), separator, phoneme) if len(separator) > 0 else phoneme
    return phoneme.strip(separator)


def _grapheme_to_phoneme(grapheme: str, separator: str = '', **kwargs) -> str:
    # NOTE: `grapheme` is split on new lines because `espeak` is inconsistent in it's handling of
    # new lines.
    return_ = (separator + '\n' + separator).join([
        _grapheme_to_phoneme_helper(s, separator=separator, **kwargs) for s in grapheme.split('\n')
    ])
    # NOTE: We need to remove double separators from when there are consecutive new lines like
    # "\n\n\n", for example.
    if len(separator) > 0:
        return_ = re.sub(r'%s+' % re.escape(separator), separator, return_).strip(separator)
    return return_


def grapheme_to_phoneme(graphemes: typing.List[str],
                        chunk_size: int = 128,
                        **kwargs) -> typing.List[str]:
    """ Convert graphemes into phonemes without perserving punctuation or white-spaces.

    NOTE: `espeak` can give different results for the same argument, sometimes. For example,
    "Fitness that's invigorating, not intimidating!" sometimes returns...
    1. "f|ˈ|ɪ|t|n|ə|s| |ð|æ|t|s| |ɪ|n|v|ˈ|ɪ|ɡ|ɚ|ɹ|ˌ|eɪ|ɾ|ɪ|ŋ|,| "...
    2. "f|ˈ|ɪ|t|n|ə|s| |ð|æ|t|s| |ɪ|n|v|ˈ|ɪ|ɡ|oː|ɹ|ˌ|eɪ|ɾ|ɪ|ŋ|,| "...

    TODO: Replace the eSpeak with a in-house solution including:
    - `lib.text.normalize_non_standard_words`
    - CMU dictionary or https://github.com/kylebgorman/wikipron for most words
    - spaCy for homographs similar to https://github.com/Kyubyong/g2p
    - A neural network trained on CMU dictionary for words not in the dictionaries.

    Args:
        graphemes: The graphemes to convert to phonemes.
        chunk_size: `chunk_size` parameter passed to `imap` for multiprocessing.
        **kwargs: Key-word arguments passed to `_grapheme_to_phoneme`.
    """
    part = partial(_grapheme_to_phoneme, **kwargs)
    if len(graphemes) < chunk_size:
        return [part(g) for g in graphemes]

    logger.info('Getting phonemes for %d graphemes.', len(graphemes))
    with lib.utils.Pool(1 if lib.environment.IS_TESTING_ENVIRONMENT else os.cpu_count()) as pool:
        return list(tqdm(pool.imap(part, graphemes, chunksize=chunk_size), total=len(graphemes)))


def natural_keys(text: str) -> typing.List[typing.Union[str, int]]:
    """ Returns keys (`list`) for sorting in a "natural" order.

    Inspired by: http://nedbatchelder.com/blog/200712/human_sorting.html
    """
    return [(int(char) if char.isdigit() else char) for char in re.split(r'(\d+)', str(text))]


def strip(text: str) -> typing.Tuple[str, str, str]:
    """ Strip and return the stripped text.

    Returns:
        text: The stripped text.
        left: Text stripped from the left-side.
        right: Text stripped from the right-side.
    """
    input_ = text
    text = text.rstrip()
    right = input_[len(text):]
    text = text.lstrip()
    left = input_[:len(input_) - len(right) - len(text)]
    return text, left, right


def normalize_vo_script(text: str, strip: bool = True) -> str:
    """ Normalize a voice-over script such that only readable characters remain.

    References:
    - Generic package for text cleaning: https://github.com/jfilter/clean-text
    - ASCII characters: https://www.ascii-code.com/
    - `Unidecode` vs `unicodedata`:
      https://stackoverflow.com/questions/517923/what-is-the-best-way-to-remove-accents-normalize-in-a-python-unicode-string
    """
    text = str(text)
    text = ftfy.fix_text(text)
    text = text.replace('\f', '\n')
    text = text.replace('\t', '  ')
    text = unidecode.unidecode(text)
    if strip:
        text = text.strip()
    return text


_READABLE_CHARACTERS = set(
    lib.utils.flatten(normalize_vo_script(chr(i), strip=False) for i in range(0, 128)))


def is_normalized_vo_script(text: str) -> bool:
    """ Return `True` if `text` has been normalized to a small set of characters. """
    return len(set(text) - _READABLE_CHARACTERS) == 0


@lru_cache(maxsize=None)
def _nltk_download(dependency):
    """ Run `nltk.download` but only once per process. """
    nltk.download(dependency)


@lru_cache(maxsize=None)
def load_en_core_web_md(*args, **kwargs) -> spacy.lang.en.English:
    """ Load and cache in memory a spaCy `spacy.lang.en.English` object. """
    return en_core_web_md.load(*args, **kwargs)


@lru_cache(maxsize=None)
def load_en_english(*args, **kwargs) -> spacy.lang.en.English:
    """ Load and cache in memory a spaCy `spacy.lang.en.English` object. """
    return spacy_en.English()


def normalize_non_standard_words(text: str, variety: str = 'AmE', **kwargs) -> str:
    """ Noramlize non-standard words (NSWs) into standard words.

    References:
      - Text Normalization Researcher, Richard Sproat:
        https://scholar.google.com/citations?hl=en&user=LNDGglkAAAAJ&view_op=list_works&sortby=pubdate
        https://rws.xoba.com/
      - Timeline:
        - Sproat & Jaitly Dataset (2020):
          https://www.kaggle.com/richardwilliamsproat/text-normalization-for-english-russian-and-polish
        - Zhang & Sproat Paper (2019):
          https://www.mitpressjournals.org/doi/full/10.1162/COLI_a_00349
        - Wu & Gorman & Sproat Code (2016):
            https://github.com/google/TextNormalizationCoveringGrammars
        - Ford & Flint `normalise` Paper (2017): https://www.aclweb.org/anthology/W17-4414.pdf
        - Ford & Flint `normalise` Code (2017): https://github.com/EFord36/normalise
        - Sproat & Jaitly Dataset (2017): https://github.com/rwsproat/text-normalization-data
        - Siri (2017): https://machinelearning.apple.com/research/inverse-text-normal
        - Sproat Kaggle Challenge (2017):
          https://www.kaggle.com/c/text-normalization-challenge-english-language/overview
        - Sproat Kaggle Dataset (2017): https://www.kaggle.com/google-nlu/text-normalization
        - Sproat TTS Tutorial (2016): https://github.com/rwsproat/tts-tutorial
        - Sproat & Jaitly Paper (2016): https://arxiv.org/pdf/1611.00068.pdf
        - Wu & Gorman & Sproat Paper (2016): https://arxiv.org/abs/1609.06649
        - Gorman & Sproat Paper (2016): https://transacl.org/ojs/index.php/tacl/article/view/897/213
        - Ebden and Sproat (2014) Code:
          https://github.com/google/sparrowhawk
          https://opensource.google/projects/sparrowhawk
          https://www.kaggle.com/c/text-normalization-challenge-english-language/discussion/39061#219939
        - Sproat Course (2011):
          https://web.archive.org/web/20181029032542/http://www.csee.ogi.edu/~sproatr/Courses/TextNorm/
      - Other:
        - MaryTTS text normalization:
          https://github.com/marytts/marytts/blob/master/marytts-languages/marytts-lang-en/src/main/java/marytts/language/en/Preprocess.java
        - ESPnet text normalization:
          https://github.com/espnet/espnet_tts_frontend/tree/master/tacotron_cleaner
        - Quora question on text normalization:
          https://www.quora.com/Is-it-possible-to-use-festival-toolkit-for-text-normalization
        - spaCy entity classification:
          https://explosion.ai/demos/displacy-ent
          https://prodi.gy/docs/named-entity-recognition#manual-model
          https://spacy.io/usage/examples#training
        - Dockerized installation of festival by Google:
          https://github.com/google/voice-builder

    TODO:
       - Following the state-of-the-art approach presented here:
         https://www.kaggle.com/c/text-normalization-challenge-english-language/discussion/43963
         Use spaCy to classify entities, and then use a formatter to clean up the strings. The
         dataset was open-sourced here:
         https://www.kaggle.com/richardwilliamsproat/text-normalization-for-english-russian-and-polish
         A formatter can be found here:
         https://www.kaggle.com/neerjad/class-wise-regex-functions-l-b-0-995
         We may need to train spaCy to detect new entities, if the ones already supported are not
         enough via prodi.gy:
         https://prodi.gy/docs/named-entity-recognition#manual-model
       - Adopt Google's commercial "sparrowhawk" or the latest grammar
         "TextNormalizationCoveringGrammars" for text normalization.
    """
    for dependency in ('brown', 'names', 'wordnet', 'averaged_perceptron_tagger',
                       'universal_tagset'):
        _nltk_download(dependency)

    tokens = [[t.text, t.whitespace_] for t in load_en_english()(text)]
    merged = [tokens[0]]
    # TODO: Use https://spacy.io/usage/linguistic-features#retokenization
    for token, whitespace in tokens[1:]:
        # NOTE: For example, spaCy tokenizes "$29.95" as two tokens, and this undos that.
        if (merged[-1][0] == '$' or token == '$') and merged[-1][1] == '':
            merged[-1][0] += token
            merged[-1][1] = whitespace
        else:
            merged.append([token, whitespace])

    assert ''.join(lib.utils.flatten(merged)) == text
    normalized = normalise.normalise([t[0] for t in merged], variety=variety, **kwargs)
    return ''.join(lib.utils.flatten([(n.strip(), m[1]) for n, m in zip(normalized, merged)]))


def format_alignment(tokens: typing.List[str], other_tokens: typing.List[str],
                     alignments: typing.List[typing.Tuple[int, int]]) -> typing.Tuple[str, str]:
    """ Format strings to be printed of the alignment.

    Example:

        >>> tokens = ['i','n','t','e','n','t','i','o','n']
        >>> other_tokens = ['e','x','e','c','u','t','i','o','n']
        >>> alignment = [(0, 0), (2, 1), (3, 2), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8)]
        >>> formatted = format_alignment(tokens, other_tokens, alignment)
        >>> formatted[0]
        'i n t e   n t i o n'
        >>> formatted[1]
        'e   x e c u t i o n'

    Args:
        tokens: Sequences of strings.
        other_tokens: Sequences of strings.
        alignment: Alignment between the tokens in both sequences. The alignment is a sequence of
            sorted tuples with a `tokens` index and `other_tokens` index.

    Returns:
        formatted_tokens: String formatted for printing including `tokens`.
        formatted_other_tokens: String formatted for printing including `other_tokens`.
    """
    tokens_index = 0
    other_tokens_index = 0
    alignments_index = 0
    tokens_string = ''
    other_tokens_string = ''
    while True and len(alignments) != 0:
        alignment = alignments[alignments_index]
        other_token = other_tokens[other_tokens_index]
        token = tokens[tokens_index]
        if tokens_index == alignment[0] and other_tokens_index == alignment[1]:
            padded_format = '{:' + str(max(len(token), len(other_token))) + 's} '
            tokens_string += padded_format.format(token)
            other_tokens_string += padded_format.format(other_token)
            tokens_index += 1
            other_tokens_index += 1
            alignments_index += 1
        elif tokens_index == alignment[0]:
            other_tokens_string += other_token + ' '
            tokens_string += ' ' * (len(other_token) + 1)
            other_tokens_index += 1
        elif other_tokens_index == alignment[1]:
            tokens_string += token + ' '
            other_tokens_string += ' ' * (len(token) + 1)
            tokens_index += 1

        if alignments_index == len(alignments):
            break

    return tokens_string.rstrip(), other_tokens_string.rstrip()


def _is_in_window(value: int, window: typing.Tuple[int, int]) -> bool:
    """ Check if `value` is in the range [`window[0]`, `window[1]`).  """
    return value >= window[0] and value < window[1]


def align_tokens(
    tokens: typing.Union[typing.List[str], str],
    other_tokens: typing.Union[typing.List[str], str],
    window_length: typing.Optional[int] = None,
    all_alignments: bool = False,
    allow_substitution: typing.Callable[[str, str], bool] = lambda a, b: True
) -> typing.Tuple[int, typing.List[typing.Tuple[int, int]]]:
    """ Compute the alignment between `tokens` and `other_tokens`.

    Base algorithm implementation: https://en.wikipedia.org/wiki/Levenshtein_distance

    This implements a modified version of the levenshtein distance algorithm. The modifications are
    as follows:
      - We do not assume each token is a character; therefore, the cost of substition is the
        levenshtein distance between tokens. The cost of deletion or insertion is the length
        of the token. In the case that the tokens are characters, this algorithms functions
        equivalently to the original.
      - The user may specify a `window_length`. Given that the window length is the same length as
        the smallest sequence, then this algorithm functions equivalently to the original;
        otherwise, not every token in both sequences is compared to compute the alignment. A user
        would use this parameter if the two sequences are mostly aligned. The runtime of the
        algorithm with a window length is
        O(`window_length` * `max(len(tokens), len(other_tokens))`).

    Args:
        tokens: Sequences of strings.
        other_tokens: Sequences of strings.
        window_length: Approximately the maximum number of consecutive insertions or deletions
            required to align two similar sequences.
        allow_substitution: Callable that returns `True` if the substitution is allowed between two
            tokens `a` and `b`.

    Returns:
        cost: The cost of alignment.
        alignment: The alignment consisting of a sequence of indicies.
    """
    # For `window_center` to be on a diagonal with an appropriate slope to align both sequences,
    # `tokens` needs to be the longer sequence.
    flipped = len(other_tokens) > len(tokens)
    if flipped:
        other_tokens, tokens = (tokens, other_tokens)

    if window_length is None:
        window_length = len(other_tokens)

    alignment_window_length = min(2 * window_length + 1, len(other_tokens) + 1)
    row_one: typing.List[typing.Optional[int]] = [None] * alignment_window_length
    row_two: typing.List[typing.Optional[int]] = [None] * alignment_window_length
    # NOTE: This operation copies a reference to the initial list `alignment_window_length` times.
    # Example:
    # >>> list_of_lists = [[]] * 10
    # >>> list_of_lists[0].append(1)
    # >>> list_of_lists
    # [[1], [1], [1], [1], [1], [1], [1], [1], [1], [1]]
    row_one_paths: typing.List[typing.List[typing.List[typing.Tuple[int, int]]]]
    row_one_paths = [[[]]] * alignment_window_length
    row_two_paths: typing.List[typing.List[typing.List[typing.Tuple[int, int]]]]
    row_two_paths = [[[]]] * alignment_window_length

    # Number of edits to convert a substring of the `other_tokens` into the empty string.
    row_one[0] = 0
    for i in range(min(window_length, len(other_tokens))):
        assert row_one[i] is not None
        row_one[i + 1] = typing.cast(int, row_one[i]) + len(
            other_tokens[i])  # Deletion of `other_tokens[:i + 1]`.

    # Both `row_one_window` and `row_two_window` are not inclusive of the maximum.
    row_one_window = (0, min(len(other_tokens) + 1, window_length + 1))

    for i in tqdm(range(len(tokens))):
        # TODO: Consider setting the `window_center` at the minimum index in `row_one`. There are
        # a number of considerations to make:
        # 1. The window no longer guarantees that the last window will have completed both
        # sequences.
        # 2. Smaller indicies in `row_one` have not completed as much of the sequences as larger
        # sequences in `row_one`.
        window_center = min(i, len(other_tokens))
        row_two_window = (max(0, window_center - window_length),
                          min(len(other_tokens) + 1, window_center + window_length + 1))
        if ((row_two_window[1] - row_two_window[0]) < len(row_two) and
                row_two_window[1] == len(other_tokens) + 1):
            row_two = row_two[:(row_two_window[1] - row_two_window[0])]
            row_two_paths = row_two_paths[:(row_two_window[1] - row_two_window[0])]

        for j in range(*row_two_window):
            choices = []

            if _is_in_window(j, row_one_window):
                deletion_cost = typing.cast(int, row_one[j - row_one_window[0]]) + len(tokens[i])
                deletion_path = row_one_paths[j - row_one_window[0]]
                choices.append((deletion_cost, deletion_path))
            if _is_in_window(j - 1, row_two_window):
                assert row_two[j - 1 - row_two_window[0]] is not None
                insertion_cost = typing.cast(int, row_two[j - 1 - row_two_window[0]]) + len(
                    other_tokens[j - 1])
                insertion_path = row_two_paths[j - 1 - row_two_window[0]]
                choices.append((insertion_cost, insertion_path))
            if _is_in_window(j - 1, row_one_window):
                token = tokens[i]
                other_token = other_tokens[j - 1]
                if token == other_token or allow_substitution(token, other_token):
                    substition_cost = Levenshtein.distance(token, other_token)
                    substition_cost = row_one[j - 1 - row_one_window[0]] + substition_cost
                    alignment = (j - 1, i) if flipped else (i, j - 1)
                    substition_path = row_one_paths[j - 1 - row_one_window[0]]
                    substition_path = [p + [alignment] for p in substition_path]
                    choices.append((substition_cost, substition_path))

            # NOTE: `min` picks the first occurring minimum item in the iterable passed.
            # NOTE: Substition is put last on purpose to discourage substition if other options are
            # available that cost the same.
            min_cost, min_paths = min(choices, key=lambda p: p[0])

            row_two[j - row_two_window[0]] = min_cost
            row_two_paths[j - row_two_window[0]] = min_paths

        row_one = row_two[:]
        row_one_paths = row_two_paths[:]
        row_one_window = row_two_window

    return (
        typing.cast(int, row_one[-1]),
        row_one_paths[-1][0],
    )

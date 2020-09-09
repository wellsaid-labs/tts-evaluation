from functools import lru_cache
from functools import partial

import logging
import os
import re
import shlex
import subprocess

from third_party import LazyLoader
from tqdm import tqdm

import en_core_web_md
import ftfy
import nltk
import unidecode
Levenshtein = LazyLoader('Levenshtein', globals(), 'Levenshtein')

from src.environment import IS_TESTING_ENVIRONMENT
from src.utils import flatten
from src.utils import Pool

logger = logging.getLogger(__name__)


def _grapheme_to_phoneme_helper(grapheme,
                                service='espeak',
                                flags=('--ipa=3', '-q', '-ven-us', '--stdin'),
                                separator='',
                                service_separator='_'):
    """
    Args:
        grapheme (str)
        service (str, optional): The service used to compute phonemes.
        flags (list of str, optional): The list of flags to add to the service.
        separator (str, optional): The separator used to separate phonemes.
        service_separator (str, optional): The separator used by the service between phonemes.

    Returns:
        phoneme (str)
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


def _grapheme_to_phoneme(grapheme, separator='', **kwargs):
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


def grapheme_to_phoneme(graphemes, chunk_size=128, **kwargs):
    """ Convert graphemes into phonemes without perserving punctuation.

    NOTE: `espeak` can give different results for the same argument, sometimes. For example,
    "Fitness that's invigorating, not intimidating!" sometimes returns...
    1. "f|ˈ|ɪ|t|n|ə|s| |ð|æ|t|s| |ɪ|n|v|ˈ|ɪ|ɡ|ɚ|ɹ|ˌ|eɪ|ɾ|ɪ|ŋ|,| "...
    2. "f|ˈ|ɪ|t|n|ə|s| |ð|æ|t|s| |ɪ|n|v|ˈ|ɪ|ɡ|oː|ɹ|ˌ|eɪ|ɾ|ɪ|ŋ|,| "...

    TODO: Instead of using eSpeak, we could use a combination of `normalise`, CMU's dictionary,
    spaCy, and a small NN for new words.

    Args:
        graphemes (str): The graphemes to convert to phonemes.
        chunk_size (int, optional): `chunk_size` parameter passed to `imap`.
        **kwargs: Key-word arguments passed to `_grapheme_to_phoneme`.

    Returns:
        phoneme (str)
    """
    part = partial(_grapheme_to_phoneme, **kwargs)
    if len(graphemes) < chunk_size:
        return [part(g) for g in graphemes]

    logger.info('Getting phonemes for %d graphemes.', len(graphemes))
    with Pool(1 if IS_TESTING_ENVIRONMENT else os.cpu_count()) as pool:
        return list(tqdm(pool.imap(part, graphemes, chunksize=chunk_size), total=len(graphemes)))


def natural_keys(text):
    """ Returns keys (`list`) for sorting in a "natural" order.
    Inspired by: http://nedbatchelder.com/blog/200712/human_sorting.html
    """
    return [(int(char) if char.isdigit() else char) for char in re.split(r'(\d+)', str(text))]


def strip(text):
    """ Strip and return the stripped text.

    Args:
        text (str)

    Returns:
        (str): The stripped text.
        (str): Text stripped from the left.
        (str): Text stripped from the right.
    """
    input_ = text
    text = text.rstrip()
    right = input_[len(text):]
    text = text.lstrip()
    left = input_[:len(input_) - len(right) - len(text)]
    return text, left, right


def normalize_vo_script(text):
    """ Normalize a voice-over script such that only readable characters remain.

    References:
    - Generic package for text cleaning: https://github.com/jfilter/clean-text
    - ASCII characters: https://www.ascii-code.com/
    - `Unidecode` vs `unicodedata`:
      https://stackoverflow.com/questions/517923/what-is-the-best-way-to-remove-accents-normalize-in-a-python-unicode-string

    TODO: Test control characters that produce whitespace (U+09, U+0A, U+0C, U+0D) are handled.
    TODO: Test all other control characters are handled correctly.
    """
    text = text.encode().decode('unicode-escape')
    text = ftfy.fix_text(text)
    text = text.replace('\f', '\n')
    text = text.replace('\t', '  ')
    return unidecode.unidecode(text).strip()


_READABLE_CHARACTERS = set(
    normalize_vo_script(chr(i)) if i in [10, 32] else chr(i) for i in range(0, 128))


def is_normalized_vo_script(text):
    """ Return `True` if `text` has been normalized to a small set of characters.
    """
    return len(set(text) - _READABLE_CHARACTERS) == 0


def normalize_non_standard_words(text, variety='AmE'):
    """ Noramlize non-standard words (NSWs) into standard words.

    References:
       - Ford & Flint `normalise` Paper (2017): https://www.aclweb.org/anthology/W17-4414.pdf
       - Ford & Flint `normalise` Code (2017): https://github.com/EFord36/normalise
       - Sproat & Jaitly Dataset (2017): https://github.com/rwsproat/text-normalization-data
       - Siri (2017): https://machinelearning.apple.com/research/inverse-text-normal
       - Sproat Kaggle Challenge (2017):
         https://www.kaggle.com/c/text-normalization-challenge-english-language/overview
       - Sproat Kaggle Dataset (2017): https://www.kaggle.com/google-nlu/text-normalization
       - Sproat & Jaitly Paper (2016): https://arxiv.org/pdf/1611.00068.pdf
       - Wu & Gorman & Sproat Paper (2016): https://arxiv.org/abs/1609.06649
       - Gorman & Sproat Paper (2016): https://transacl.org/ojs/index.php/tacl/article/view/897/213
       - Ebden and Sproat (2014) Code: https://opensource.google/projects/sparrowhawk
         https://www.kaggle.com/c/text-normalization-challenge-english-language/discussion/39061#219939

    TODO: There are a number of interesting kernels and datasets hosted on Kaggle with regard
    to this problem; however, the licensing on them is vague.
    """
    for dependency in ('brown', 'names', 'wordnet', 'averaged_perceptron_tagger',
                       'universal_tagset'):
        nltk.download(dependency)
    from normalise import normalise
    return normalise(text, variety=variety)


@lru_cache(maxsize=None)
def load_en_core_web_md(*args, **kwargs):
    """ Load and cache in memory a spaCy `Language` object.
    """
    return en_core_web_md.load(*args, **kwargs)


def format_alignment(tokens, other_tokens, alignments):
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
        tokens (list or str): Sequences of strings.
        other_tokens (list or str): Sequences of strings.
        alignment (list): Alignment between the tokens in both sequences. The alignment is a
            sequence of sorted tuples with a `tokens` index and `other_tokens` index.

    Returns:
        formatted_tokens (str): String formatted for printing.
        formatted_other_tokens (str): String formatted for printing.
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


def _is_in_window(value, window):
    """ Check if `value` is in the range [`window[0]`, `window[1]`)
    """
    return value >= window[0] and value < window[1]


def align_tokens(tokens,
                 other_tokens,
                 window_length=None,
                 all_alignments=False,
                 allow_substitution=lambda a, b: True):
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
        tokens (list or str): Sequences of strings.
        other_tokens (list or str): Sequences of strings.
        window_length (int or None, optional): Approximately the maximum number of consecutive
            insertions or deletions required to align two similar sequences.
        all_alignments (bool, optional): If `True` return all optimal alignments, otherwise, return
            one optimal alignment.
        allow_substitution (callable, optional): Return a `bool` if the substitution is allowed
            between two tokens `a` and `b`.

    Returns:
        (int): The cost of alignment.
        (list): The alignment consisting of a sequence of indicies.
    """
    # For `window_center` to be on a diagonal with an appropriate slope to align both sequences,
    # `tokens` needs to be the longer sequence.
    flipped = len(other_tokens) > len(tokens)
    if flipped:
        other_tokens, tokens = (tokens, other_tokens)

    if window_length is None:
        window_length = len(other_tokens)

    alignment_window_length = min(2 * window_length + 1, len(other_tokens) + 1)
    row_one = [None] * alignment_window_length
    row_two = [None] * alignment_window_length
    # NOTE: This operation copies a reference to the initial list `alignment_window_length` times.
    # Example:
    # >>> list_of_lists = [[]] * 10
    # >>> list_of_lists[0].append(1)
    # >>> list_of_lists
    # [[1], [1], [1], [1], [1], [1], [1], [1], [1], [1]]
    row_one_paths = [[[]]] * alignment_window_length
    row_two_paths = [[[]]] * alignment_window_length

    # Number of edits to convert a substring of the `other_tokens` into the empty string.
    row_one[0] = 0
    for i in range(min(window_length, len(other_tokens))):
        row_one[i + 1] = row_one[i] + len(other_tokens[i])  # Deletion of `other_tokens[:i + 1]`.

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
                deletion_cost = row_one[j - row_one_window[0]] + len(tokens[i])
                deletion_path = row_one_paths[j - row_one_window[0]]
                choices.append((deletion_cost, deletion_path))
            if _is_in_window(j - 1, row_two_window):
                insertion_cost = row_two[j - 1 - row_two_window[0]] + len(other_tokens[j - 1])
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
            if all_alignments:
                min_paths = flatten([path for cost, path in choices if cost == min_cost])

            row_two[j - row_two_window[0]] = min_cost
            row_two_paths[j - row_two_window[0]] = min_paths

        row_one = row_two[:]
        row_one_paths = row_two_paths[:]
        row_one_window = row_two_window

    return row_one[-1], row_one_paths[-1] if all_alignments else row_one_paths[-1][0]

from Levenshtein import distance
from tqdm import tqdm

from src.utils.utils import flatten


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
                substition_cost = distance(token, other_token)
                if allow_substitution(token, other_token):
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

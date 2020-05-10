from functools import lru_cache
from functools import partial
from itertools import groupby

import logging
import os
import re
import shlex
import subprocess

from tqdm import tqdm

import en_core_web_sm

from hparams import configurable
from hparams import HParam
from src.environment import IS_TESTING_ENVIRONMENT
from src.utils import disk_cache
from src.utils import log_runtime
from src.utils import Pool
from src.utils import strip
from src.utils.disk_cache_ import make_arg_key

logger = logging.getLogger(__name__)


def grapheme_to_phoneme(grapheme, separator='', **kwargs):
    """ Convert graphemes into phonemes without perserving punctuation.

    NOTE: `grapheme` is split on new lines because `espeak` is inconsistent in it's handling of new
    lines.

    Args:
        grapheme (str): The graphemes to convert to phonemes.
        service (str, optional): The service used to compute phonemes.
        flags (list of str, optional): The list of flags to add to the service.
        separator (str, optional): The separator used to separate phonemes.
        service_separator (str, optional): The separator used by the service between phonemes.

    Returns:
        phoneme (str)
    """
    return_ = (separator + '\n' + separator).join([
        _grapheme_to_phoneme_helper(s, separator=separator, **kwargs) for s in grapheme.split('\n')
    ])
    # NOTE: We need to remove double separators from when there are consecutive new lines like
    # "\n\n\n", for example.
    if len(separator) > 0:
        return_ = re.sub(r'%s+' % re.escape(separator), separator, return_).strip(separator)
    return return_


@disk_cache
def _grapheme_to_phoneme_helper(grapheme,
                                service='espeak',
                                flags=('--ipa=3', '-q', '-ven-us', '--stdin'),
                                separator='',
                                service_separator='_'):
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


@lru_cache()
def _get_spacy_model():
    """ Get spaCy object used to tokenize and POS tag text. """
    logger.info('Loading spaCy model...')
    return en_core_web_sm.load(disable=['parser', 'ner'])


# See https://spacy.io/api/annotation#pos-tagging for all available tags.
_SPACY_PUNCT_TAG = 'PUNCT'


@disk_cache
def grapheme_to_phoneme_perserve_punctuation(text, separator='', **kwargs):
    """ Convert grapheme to phoneme while perserving punctuation.

    Args:
        text (str): Graphemes.
        separator (str): The separator used to separate phonemes, stress, and punctuation.
        **kwargs: Key-word arguments passed to `grapheme_to_phoneme`.

    Returns:
        (str): Phonemes with the original punctuation (as defined by spaCy).
    """
    tokens = _get_spacy_model()(text)

    assert len(tokens) > 0, 'Zero tokens were found in text: %s' % text
    assert text == ''.join(t.text_with_ws for t in tokens), 'Detokenization failed: %s' % text
    assert not separator or separator not in text, 'The separator is not unique.'

    # NOTE: `is_punct` is not contextual while `pos_ == _SPACY_PUNCT_TAG` is, see:
    # https://github.com/explosion/spaCy/issues/998. This enables us to phonemize cases like:
    # - "form of non-linguistic representations"  (ADJ)
    # - "The psychopaths' empathic reaction"  (PART)
    # - "judgement, name & face memory" (CCONJ)
    # - "to public interest/national security" (SYM)
    # - "spectacular, grand // desco da" (SYM)
    return_ = []
    for is_punct, group in groupby(tokens, lambda t: t.pos_ == _SPACY_PUNCT_TAG):
        phrase = ''.join([t.text_with_ws for t in group])
        is_alpha_numeric = any(c.isalpha() or c.isdigit() for c in phrase)
        if is_punct and is_alpha_numeric:
            logger.warning('Punctuation contains alphanumeric characters: %s' % phrase)
        if is_punct and not is_alpha_numeric:
            return_.extend(list(phrase))
        else:
            return_.append(grapheme_to_phoneme(phrase, separator=separator, **kwargs))
    return separator.join([t for t in return_ if len(t) > 0])


@log_runtime
@configurable
def cache_grapheme_to_phoneme_perserve_punctuation(texts,
                                                   chunksize=128,
                                                   delimiter=HParam(),
                                                   **kwargs):
    """ Batch process and cache the results for `texts` passed to
    `grapheme_to_phoneme_perserve_punctuation`.

    Args:
        texts (list of str)
        chunksize (int): `chunksize` parameter passed to `imap`.
        **kwargs: Key-word arguments passed to `grapheme_to_phoneme_perserve_punctuation`.
    """
    function = grapheme_to_phoneme_perserve_punctuation.__wrapped__
    texts = [
        t for t in texts if make_arg_key(function, t, separator=delimiter, **kwargs) not in
        grapheme_to_phoneme_perserve_punctuation.disk_cache
    ]
    if len(texts) == 0:
        return

    logger.info('Caching `grapheme_to_phoneme_perserve_punctuation` %d texts.', len(texts))
    partial_ = partial(grapheme_to_phoneme_perserve_punctuation, separator=delimiter, **kwargs)
    with Pool(1 if IS_TESTING_ENVIRONMENT else os.cpu_count()) as pool:
        iterator = zip(texts, pool.imap(partial_, texts, chunksize=chunksize))
        for text, result in tqdm(iterator, total=len(texts)):
            arg_key = make_arg_key(function, text, separator=delimiter, **kwargs)
            # NOTE: `espeak` can give different results for the same argument, sometimes. For
            # example, "Fitness that's invigorating, not intimidating!" sometimes returns...
            # 1. "f|ˈ|ɪ|t|n|ə|s| |ð|æ|t|s| |ɪ|n|v|ˈ|ɪ|ɡ|ɚ|ɹ|ˌ|eɪ|ɾ|ɪ|ŋ|,| "...
            # 2. "f|ˈ|ɪ|t|n|ə|s| |ð|æ|t|s| |ɪ|n|v|ˈ|ɪ|ɡ|oː|ɹ|ˌ|eɪ|ɾ|ɪ|ŋ|,| "...
            if arg_key not in grapheme_to_phoneme_perserve_punctuation.disk_cache:
                grapheme_to_phoneme_perserve_punctuation.disk_cache.set(arg_key, result)
            else:  # TODO: Add test case to replicate this behavior.
                cached = grapheme_to_phoneme_perserve_punctuation.disk_cache.get(arg_key)
                if cached != result:
                    logger.warning(
                        'Given `%s` `grapheme_to_phoneme_perserve_punctuation` returned '
                        'both `%s` and `%s`.', arg_key, cached, result)

    grapheme_to_phoneme_perserve_punctuation.disk_cache.save()

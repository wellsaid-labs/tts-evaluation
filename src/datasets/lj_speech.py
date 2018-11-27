from functools import partial
from pathlib import Path

import csv
import logging
import re
import unidecode

from num2words import num2words
from torchnlp.datasets import Dataset

import pandas

from src.datasets.constants import Speaker
from src.datasets.process import compute_spectrogram
from src.datasets.process import download_file_maybe_extract
from src.datasets.process import normalize_audio
from src.datasets.process import process_in_parallel
from src.datasets.process import split_dataset
from src.hparams import configurable
from src.utils import Checkpoint

logger = logging.getLogger(__name__)


def _processing_func(row,
                     directory,
                     extracted_name,
                     spectrogram_model_checkpoint_path,
                     kwargs,
                     metadata_audio_column=0,
                     metadata_audio_path_template='wavs/{}.wav',
                     metadata_text_column=1,
                     verbalize=True,
                     speaker=Speaker.LINDA_JOHNSON):  # pragma: no cover
    """
    Note:
        - ``# pragma: no cover`` is used because this functionality is run with multiprocessing
          that is not compatible with the coverage module.

    Args:
        directory (str or Path, optional): Directory to cache the dataset.
        extracted_name (str, optional): Name of the extracted dataset directory.
        spectrogram_model_checkpoint_path (str or None, optional): Spectrogram model to predict a
            ground truth aligned spectrogram.
        kwargs: Arguments passed to ``normalize_audio`` to preprocess the dataset audio.
        metadata_audio_column (int, optional): Column name or index with the audio filename.
        metadata_audio_path_template (str, optional): Given the audio column, this template
            determines the filename.
        metadata_text_column (int, optional): Column name or index with the audio transcript.
        verbalize (bool, optional): Verbalize the text.
        speaker (src.datasets.Speaker, optional)

    Returns:
        {
            text (str)
            audio_path (Path)
            spectrogram_path (Path)
            predicted_spectrogram_path (Path)
            speaker (src.datasets.Speaker)
        }
    """
    spectrogram_model_checkpoint = Checkpoint.from_path(spectrogram_model_checkpoint_path)
    text = row[metadata_text_column].strip()
    audio_path = Path(directory, extracted_name,
                      metadata_audio_path_template.format(row[metadata_audio_column]))
    text = _normalize_whitespace(text)
    text = _normalize_quotations(text)

    if verbalize:
        text = _verbalize_special_cases(audio_path, text)
        text = _expand_abbreviations(text)
        text = _verbalize_time_of_day(text)
        text = _verbalize_ordinals(text)
        text = _verbalize_currency(text)
        text = _verbalize_serial_numbers(text)
        text = _verbalize_year(text)
        text = _verbalize_numeral(text)
        text = _verbalize_number(text)
        text = _verbalize_roman_number(text)

    # NOTE: Messes up pound sign (£); therefore, this is after ``_verbalize_currency``
    text = _remove_accents(text)
    audio_path = normalize_audio(audio_path, **kwargs)
    aligned_audio_path, spectrogram_path, predicted_spectrogram_path = compute_spectrogram(
        audio_path, text, speaker, spectrogram_model_checkpoint)

    return {
        'text': text,
        'audio_path': audio_path,
        'aligned_audio_path': aligned_audio_path,
        'spectrogram_path': spectrogram_path,
        'predicted_spectrogram_path': predicted_spectrogram_path,
        'speaker': speaker,
    }


@configurable
def lj_speech_dataset(directory='data/',
                      extracted_name='LJSpeech-1.1',
                      url='http://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2',
                      check_files=['LJSpeech-1.1/metadata.csv'],
                      metadata_filename='metadata.csv',
                      metadata_quoting=csv.QUOTE_NONE,
                      metadata_delimiter='|',
                      metadata_header=None,
                      splits=(.8, .2),
                      spectrogram_model_checkpoint_path=None,
                      **kwargs):
    """ Load the Linda Johnson (LJ) Speech dataset.

    This is a public domain speech dataset consisting of 13,100 short audio clips of a single
    speaker reading passages from 7 non-fiction books. A transcription is provided for each clip.
    Clips vary in length from 1 to 10 seconds and have a total length of approximately 24 hours.

    The SoX resampling library is choosen rather than the best python resampler because the best
    python resampler does not have tools to deal with clipping. Clipping is when resampling
    predicts energy levels outside of the [-1, 1] range that need to be clipped.


    Reference:
        * Link to dataset source:
          https://keithito.com/LJ-Speech-Dataset/
        * Comparison of command line resampling libraries:
          http://www.mainly.me.uk/resampling/
        * Comparison of python resampling libraries:
          https://machinelearningmastery.com/resample-interpolate-time-series-data-python/

    Args:
        directory (str or Path, optional): Directory to cache the dataset.
        extracted_name (str, optional): Name of the extracted dataset directory.
        url (str, optional): URL of the dataset `tar.gz` file.
        check_files (list of str, optional): Check this file exists if the download was successful.
        metadata_filename (str, optional): The file containing audio metadata.
        metadata_quoting (int, optional): Control field quoting behavior per csv.QUOTE_* constants
            for the metadata file.
        metadata_delimiter (str, optional): Delimiter for the metadata file.
        metadata_header (bool, optional): If True, ``metadata_file`` has a header to parse.
        splits (tuple, optional): The number of splits and cardinality of dataset splits.
        spectrogram_model_checkpoint_path (str or None, optional): Spectrogram model to predict a
            ground truth aligned spectrogram.
        **kwargs: Arguments passed to ``normalize_audio`` to preprocess the dataset audio.

    Returns:
        :class:`torchnlp.datasets.Dataset`: Dataset with audio filenames and text annotations.

    Example:
        >>> import pprint # doctest: +SKIP
        >>> from src.hparams import set_hparams # doctest: +SKIP
        >>> from src.datasets import lj_speech_dataset # doctest: +SKIP
        >>> set_hparams() # doctest: +SKIP
        >>> train, dev = lj_speech_dataset() # doctest: +SKIP
        >>> pprint.pprint(train[0:2], width=80) # doctest: +SKIP
        [{'aligned_audio_path': PosixPath('data/LJSpeech-1.1/wavs/'
                                          'pad(rate(guard(norm(LJ014-0331,-.001)),24000)).npy'),
          'audio_path': PosixPath('data/LJSpeech-1.1/wavs/'
                                  'rate(guard(norm(LJ014-0331,-.001)),24000).wav'),
          'predicted_spectrogram_path': None,
          'speaker': Speaker(name='Linda Johnson', gender=FEMALE, id=4),
          'spectrogram_path': PosixPath('data/LJSpeech-1.1/wavs/spectrogram('
                                        'rate(guard(norm(LJ014-0331,-.001)),24000)).npy'),
          'text': 'Once a warrant-holder sent down a clerk to view certain goods, and '
                  'the clerk found that these goods had already a "stop" upon them, or '
                  'were pledged.'},
        {'aligned_audio_path': PosixPath('data/LJSpeech-1.1/wavs/'
                                         'pad(rate(guard(norm(LJ009-0184,-.001)),24000)).npy'),
          'audio_path': PosixPath('data/LJSpeech-1.1/wavs/'
                                  'rate(guard(norm(LJ009-0184,-.001)),24000).wav'),
          'predicted_spectrogram_path': None,
          'speaker': Speaker(name='Linda Johnson', gender=FEMALE, id=4),
          'spectrogram_path': PosixPath('data/LJSpeech-1.1/wavs/spectrogram('
                                        'rate(guard(norm(LJ009-0184,-.001)),24000)).npy'),
          'text': "Lord Ferrers' body was brought to Surgeons' Hall after execution in "
                  'his own carriage and six;'}]
    """
    logger.info('Loading LJ speech dataset')
    download_file_maybe_extract(url=url, directory=str(directory), check_files=check_files)
    metadata_path = Path(directory, extracted_name, metadata_filename)
    data = [row.to_dict() for _, row in pandas.read_csv(
        metadata_path,
        delimiter=metadata_delimiter,
        header=metadata_header,
        quoting=metadata_quoting).iterrows()]
    data = process_in_parallel(
        data,
        partial(
            _processing_func,
            directory=directory,
            extracted_name=extracted_name,
            spectrogram_model_checkpoint_path=spectrogram_model_checkpoint_path,
            kwargs=kwargs))
    splits = split_dataset(data, splits=splits)
    return tuple(Dataset(split) for split in splits)


'''
Modules for text preprocessing.

M-AILABS speech dataset mentioned: “numbers have been converted to words and some cleanup of
“foreign” characters (transliterations) have been applied.” The Tacotron 1 paper mentioned: “All
text in our datasets is spelled out. e.g., “16” is written as “sixteen”, i.e., our models are all
trained on normalized text.” LJ Speech dataset mentioned: “Normalized Transcription: transcription
with numbers, ordinals, and monetary units expanded into full words (UTF-8).” Tacotron 2 authors
mentioned: "We use verbalized data so numbers are extended. There is no need to lowercase or remove
punctuation.".
'''

# Reference: https://keithito.com/LJ-Speech-Dataset/
# List of (regular expression, replacement) pairs for abbreviations:
_abbreviations = [(re.compile('\\b%s\\.' % x[0], re.IGNORECASE), x[1]) for x in [
    ('mrs', 'misess'),
    ('mr', 'mister'),
    ('dr', 'doctor'),
    ('st', 'saint'),
    ('co', 'company'),
    ('jr', 'junior'),
    ('maj', 'major'),
    ('gen', 'general'),
    ('drs', 'doctors'),
    ('rev', 'reverend'),
    ('lt', 'lieutenant'),
    ('hon', 'honorable'),
    ('sgt', 'sergeant'),
    ('capt', 'captain'),
    ('esq', 'esquire'),
    ('ltd', 'limited'),
    ('col', 'colonel'),
    ('ft', 'fort'),
    ('messrs', 'messrs'),
    ('no', 'number'),
]]


def _match_case(source, target):
    """ Match ``source`` letter case to ``target`` letter case.

    Args:
        source (str): Reference text for the letter case.
        target (str): Target text to transfer the letter case.

    Returns:
        str: Target text with source the letter case.
    """
    if source.isupper():
        return target.upper()

    if source.islower():
        return target.lower()

    if source.istitle():
        return target.title()


def _iterate_and_replace(regex, text, replace, group=1):
    """ Match in ``text`` with ``regex`` then replace matches with callable.

    Args:
        regex (regular expression object or pattern :class:`str`): Pattern to match subtext.
        text (str): Source text to edit.
        replace (callable): Given a :class:`str` in ``text``, return a replacement.
        group (int, optional): Group to select in regex.
    Returns:
        (str): Editted text.
    """
    matches = re.finditer(regex, text)
    offset = 0
    for match in matches:
        start = match.start(group) + offset
        end = match.end(group) + offset
        replacement = replace(match.group(group))
        offset += start - end + len(replacement)
        text = text[:start] + replacement + text[end:]
    return text


def _expand_abbreviations(text):
    """ Expand abbreviations in ``text``.

    Notes:
        * The supported abbreviations can be found at: ``_abbreviations``.
        * Expanded abbreviations will maintain the same letter case as the initial abbreviation.

    Args:
        text (str): Text inwhich to expand abbreviations

    Returns:
        str: Text with expanded abbreviations.

    Example:
        >>> _expand_abbreviations('Mr. Gurney')
        'Mister Gurney'
    """
    for regex, expansion in _abbreviations:
        text = _iterate_and_replace(regex, text, lambda s: _match_case(s, expansion), group=0)
    return text


# Regular expression matching whitespace:
_whitespace_re = re.compile(r'\s+')


def _normalize_whitespace(text):
    """ Normalize white spaces in text.

    Ensure there is only one white space at a time and no white spaces at the end.

    Args:
        text (str): Text to normalize

    Returns
        str: Normalized text.

    Example:
        >>> _normalize_whitespace('Mr.     Gurney   ')
        'Mr. Gurney'
    """
    return re.sub(_whitespace_re, ' ', text).strip()


def _normalize_quotations(text):
    """ Remove accents from the text.

    Args:
        text (str): Text to normalize.

    Returns
        str: Normalized text.

    Example:
        >>> _normalize_quotations('“sponge,”')
        '"sponge,"'
    """
    text = text.replace('“', '"')
    text = text.replace('”', '"')
    text = text.replace('’', '\'')
    text = text.replace('‘', '\'')
    return text


def _remove_accents(text):
    """ Remove accents from the text.

    Args:
        text (str): Text to normalize.

    Returns
        str: Normalized text.

    Example:
        >>> _remove_accents('Málaga')
        'Malaga'
    """
    return unidecode.unidecode(text)


_special_cases = {
    'LJ044-0055': ('544 Camp Street New', 'five four four Camp Street New'),
    'LJ028-0180': ('In the year 562', 'In the year five sixty-two'),
    'LJ047-0063': ('602 Elsbeth Street', 'six oh two Elsbeth Street'),
    'LJ047-0160': ('411 Elm Street', 'four one one Elm Street'),
    'LJ047-0069': ('214 Neely Street', 'two one four Neely Street'),
    'LJ040-0121': ('P.S. 117', 'P.S. one seventeen'),
    'LJ032-0036': ('No. 2,202,130,462', 'No. two two zero two one three zero four six two'),
    'LJ029-0193': ('100 extra off-duty', 'one hundred extra off-duty'),
}

_re_filename = re.compile('LJ[0-9]{3}-[0-9]{4}')


def _verbalize_special_cases(wav, text):
    """
    Uses ``_special_cases`` to verbalize wav.

    Args:
        wav (str): Filename of the WAV file (e.g. LJ044-0055)
        text (str): Text associated with WAV file

    Returns:
        text (str): Text with special cases verbalized.
    """
    basename = wav.name[:10]  # Extract 10 characters similar to `LJ029-0193`
    assert _re_filename.match(basename)
    if basename in _special_cases:
        return text.replace(*_special_cases[basename])
    return text


_re_time_of_day = re.compile(r'([0-9]{1,2}:[0-9]{1,2})')


def _verbalize_time_of_day(text):
    """ Verbalize text.

    Args:
        text (str): Text to verbalize.

    Returns:
        text (str): Text verbalized.

    Example:
        >>> _verbalize_time_of_day('San Antonio at 1:30 p.m.,')
        'San Antonio at one thirty p.m.,'
    """

    def _replace(match):
        split = match.split(':')
        assert len(split) == 2
        words = [num2words(int(num)) for num in split]
        ret = ' '.join(words)
        return ret

    return _iterate_and_replace(_re_time_of_day, text, _replace)


_re_ordinals = re.compile(r'([0-9]+(st|nd|rd|th))')


def _verbalize_ordinals(text):
    """ Verbalize text.

    Args:
        text (str): Text to verbalize.

    Returns:
        text (str): Text verbalized.

    Example:
        >>> _verbalize_ordinals('between May 1st, 1827,')
        'between May first, 1827,'
    """

    def _replace(match):
        digit = ''.join([c for c in match if c.isdigit()])
        ret = num2words(int(digit), ordinal=True)
        return ret

    return _iterate_and_replace(_re_ordinals, text, _replace)


_re_currency = re.compile(r'(\S*([$£]{1}[0-9\,\.]+\b))')


def _verbalize_currency(text):
    """ Verbalize text.

    Args:
        text (str): Text to verbalize.

    Returns:
        text (str): Text verbalized.

    Example:
        >>> _verbalize_currency('inch BBL, unquote, cost $29.95.')
        'inch BBL, unquote, cost twenty-nine dollars, ninety-five cents.'
    """

    def _replace(match):
        digit = match[1:].replace(',', '')
        ret = num2words(digit, to='currency', currency='USD')
        ret = ret.replace(', zero cents', '')
        ret = ret.replace('hundred and', 'hundred')
        if '£' in match:
            # num2words has bugs with their GBP currency
            ret = ret.replace('dollar', 'pound')
            ret = ret.replace('cents', 'pence')
            ret = ret.replace('cent', 'penny')
        return ret

    return _iterate_and_replace(_re_currency, text, _replace)


_re_po_box = re.compile(r'([Bb]ox [0-9]+\b)')
_re_serial_number = re.compile(r'(\b[A-Za-z]+[0-9]+\b)')


def _verbalize_serial_numbers(text):
    """ Verbalize text.

    Args:
        text (str): Text to verbalize.

    Returns:
        text (str): Text verbalized.

    Example:
        >>> _verbalize_serial_numbers('Post Office Box 2915, Dallas, Texas')
        'Post Office Box two nine one five, Dallas, Texas'
        >>> _verbalize_serial_numbers('serial No. C2766, which was also found')
        'serial No. C two seven six six, which was also found'
    """

    def _replace(match):
        split = match.split(' ')
        ret = [num2words(int(t)) if t.isdigit() else t for t in list(split[-1])]
        ret = ' '.join(ret)
        if len(split) == 2:
            ret = split[0] + ' ' + ret
        return ret

    for regex in [_re_po_box, _re_serial_number]:
        text = _iterate_and_replace(regex, text, _replace)
    return text


_re_year_thousand = re.compile(r'(\b[0-9]{4}\b)')
_re_year_hundred = re.compile(r'\b(?:in|In) ([0-9]{3})\b')
_re_year_bce = re.compile(r'\b([0-9]{3}) B\.C\b')


def _verbalize_year(text):
    """ Verbalize text.

    Args:
        text (str): Text to verbalize.

    Returns:
        text (str): Text verbalized.

    Example:
        >>> _verbalize_year('Newgate down to 1818,')
        'Newgate down to eighteen eighteen,'
        >>> _verbalize_year('It was about 250 B.C., when the great')
        'It was about two fifty B.C., when the great'
        >>> _verbalize_year('In 606, Nineveh')
        'In six oh-six, Nineveh'
    """

    def _replace(match):
        ret = num2words(int(match), lang='en', to='year')
        return ret

    for regex in [_re_year_thousand, _re_year_hundred, _re_year_bce]:
        text = _iterate_and_replace(regex, text, _replace)
    return text


_re_numeral = re.compile(r'(?:Number|number) ([0-9]+)')


def _verbalize_numeral(text):
    """ Verbalize text.

    Args:
        text (str): Text to verbalize.

    Returns:
        text (str): Text verbalized.

    Example:
        >>> _verbalize_numeral(_expand_abbreviations('Exhibit No. 143 as the'))
        'Exhibit Number one forty-three as the'
    """

    def _replace(match):
        ret = num2words(int(match), lang='en', to='year')
        return ret

    return _iterate_and_replace(_re_numeral, text, _replace)


_re_roman_number = re.compile(
    r'\b(?:George|Charles|Napoleon|Henry|Nebuchadnezzar|William) ([IV]+\.{0,})')


def _verbalize_roman_number(text):
    """ Verbalize text.

    Args:
        text (str): Text to verbalize.

    Returns:
        text (str): Text verbalized.

    Example:
        >>> _verbalize_roman_number('William IV. was also the victim')
        'William the fourth was also the victim'
    """

    def _replace(match):
        if match[-1] == '.':
            match = match[:-1]

        # 0 - 9 roman number to int
        if 'V' not in match:
            num = len(match)
        elif 'IV' == match:
            num = 4
        else:
            num = 5 + len(match) - 1

        ret = 'the ' + num2words(int(num), to='ordinal')
        return ret

    return _iterate_and_replace(_re_roman_number, text, _replace)


_re_number = re.compile(r'(\b[0-9]{1}[0-9\.\,]{0,}\b)')


def _verbalize_number(text):
    """ Verbalize text.

    Args:
        text (str): Text to verbalize.

    Returns:
        text (str): Text verbalized.

    Example:
        >>> _verbalize_number('Chapter 4. The Assassin:')
        'Chapter four. The Assassin:'
        >>> _verbalize_number('was shipped on March 20, and the')
        'was shipped on March twenty, and the'
        >>> _verbalize_number('distance of 265.3 feet was, quote')
        'distance of two hundred sixty-five point three feet was, quote'
        >>> _verbalize_number('information on some 50,000 cases')
        'information on some fifty thousand cases'
        >>> _verbalize_number('PRS received items in 8,709 cases')
        'PRS received items in eight thousand, seven hundred nine cases'
    """

    def _replace(match):
        match = match.replace(',', '')
        ret = num2words(float(match))
        ret = ret.replace('hundred and', 'hundred')
        return ret

    return _iterate_and_replace(_re_number, text, _replace)

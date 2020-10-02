"""
Convert graphemes to phonemes.

Example:

    $ python -m src.bin.evaluate.grapheme_to_phoneme --grapheme 'Add your text here.'
"""
import argparse
import logging

# NOTE: Some modules log on import; therefore, we first setup logging.
from src.environment import set_basic_logging_config

set_basic_logging_config()

from src.hparams import set_hparams
from src.text import grapheme_to_phoneme_perserve_punctuation
from src.utils import get_functions_with_disk_cache

logger = logging.getLogger(__name__)

if __name__ == '__main__':  # pragma: no cover
    parser = argparse.ArgumentParser()
    parser.add_argument('--grapheme', type=str, help='Grapheme to parse.')
    args = parser.parse_args()
    set_hparams()
    for function in get_functions_with_disk_cache():  # NOTE: The cache is unnecessary in this case.
        function.use_disk_cache(False)
    logger.info('Phoneme: %s',
                grapheme_to_phoneme_perserve_punctuation(args.grapheme, separator='|'))

"""Convert graphemes to phonemes.

Example:

    $ python -m run.evaluate.grapheme_to_phoneme 'Add your text here.'
    ˈ|æ|d| |j|ʊɹ| |t|ˈ|ɛ|k|s|t| |h|ˈ|ɪɹ|.
"""
import sys

import lib

if __name__ == "__main__":  # pragma: no cover
    print(lib.text.grapheme_to_phoneme([sys.argv[1]], separator="|")[0])

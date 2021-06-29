"""Convert graphemes to phonemes.

Example:

    $ python -m run.evaluate.grapheme_to_phoneme 'Add your text here.'
    ˈ|æ|d| |j|ʊɹ| |t|ˈ|ɛ|k|s|t| |h|ˈ|ɪɹ|.
"""
import sys
import warnings

import lib

if __name__ == "__main__":  # pragma: no cover
    with warnings.catch_warnings():
        message = r".*No config for.*"
        warnings.filterwarnings("ignore", module=r".*hparams", message=message)
        print(lib.text.grapheme_to_phoneme(sys.argv[1], separator="|"))

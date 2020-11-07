import re

import typer

import lib


def main(script: str) -> str:
    """ Normalize voice-over SCRIPT such that the SCRIPT and voice-over match. """
    # TODO: Should we adopt a grammar checker to help normalize the script?
    text = lib.text.normalize_vo_script(script)
    text = text.replace('®', '')
    text = text.replace('™', '')
    # NOTE: Remove HTML tags
    text = re.sub('<.*?>', '', text)
    # NOTE: Fix for a missing space between end and beginning of a sentence. Example match is
    # deliminated with angle brackets, see here:
    #   the cold w<ar.T>he term 'business ethics'
    text = re.sub(r"([a-z]{2}[.!?])([A-Z])", r"\1 \2", text)
    return text


if __name__ == '__main__':  # pragma: no cover
    typer.run(main)

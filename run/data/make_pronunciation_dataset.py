"""
Create a pronunciation dataset using Google TTS in combination with the
American English Pronunciation Dictionary.

TODO:
- Look into commercial English Pronunciation Dictionaries like those referenced in this tutorial
  https://github.com/rwsproat/tts-tutorial
- https://github.com/nathanielove/English-words-pronunciation-mp3-audio-download. This has 30,362
  American English pronunciations from www.onelook.com and dictionary.cambridge.org. They also have
  many other sources; however, it's not clear which dialect is being used.
- We could scrape Google's pronunciation dictionary, like so:
  https://ssl.gstatic.com/dictionary/static/pronunciation/2021-03-01/audio/he/hello_en_us_1.mp3
- Use GCP's SSML feature to pronounce words a specific way in accordance to a
  Pronunciation Dictionary.
- Train on "Never Again by Doug Nufer" which is a book with 40,000 unique words based on the concept
  of https://www.mentalfloss.com/article/88172/8-extraordinary-examples-constrained-writing
- Try using the Combilex or Unisyn pronunciation dictionaries
  https://isca-speech.org/archive_v0/Interspeech_2020/pdfs/2618.pdf. "We obtain phoneme sequences
  from the Unilex pronunciation lexicon [5], as it has wider word type coverage (167,000 entries)
  and better consistency than open source lexica such as CMUdict"
- Run diacritics through Google's TTS.
- Consider, randomly, repeating some words, several times in a list, to teach the model to be
  consistent.
- GCP may, even if the word is lower case, pronounce it, as an initialism.
- Add dots in between letters as part of this dataset, also, to have more coverage over initialisms.

Other resources:
- https://nakanishi.kobegakuin-gc.jp/uploads/nakanishi/files/nakanishi/200910_Database.pdf
- https://noriko-nakanishi.com/sounds/
- 10,000+ pronunciatios http://shtooka.net/download.php
- https://opendata.stackexchange.com/questions/840/database-of-english-words-pronunciation
- https://forvo.com/

Note on Compression:
    Use ``tar -czvf name-of-archive.tar.gz /path/to/directory-or-file`` to compress the archive. For
    those using Mac OS do not use "compress" to create a `.zip` file instead [1].

[1] Mac OS uses Archive Utility to compress a directory creaing by default a
"Compressed UNIX CPIO Archive file" (CPGZ) under the `.zip` extension. The CPGZ is created with
Apple's "Apple gzip" that a Linux gzip implementations are unable to handle.
https://www.intego.com/mac-security-blog/understanding-compressed-files-and-apples-archive-utility

Usage:
    NAME="gcp_pronunciation_dictionary"
    ROOT=$(pwd)/disk/data/$NAME
    GCS_URI=gs://wellsaid_labs_datasets/
    python -m run.data.make_pronunciation_dataset --dataset-name=$NAME
    cd $ROOT
    tar -czvf "$NAME.tar.gz" metadata.csv wavs
    gsutil -m cp -r -n $NAME.tar.gz $GCS_URI
"""
import logging
import math
import multiprocessing.pool
import pathlib
import random
import re
import string
import typing
from functools import partial

import typer
from third_party import LazyLoader
from tqdm import tqdm

import run
from lib.environment import set_basic_logging_config
from lib.text import load_cmudict_syl

if typing.TYPE_CHECKING:  # pragma: no cover
    import google.auth as google_auth
    import google.auth.credentials as google_auth_credentials
    import google.cloud.texttospeech_v1beta1 as tts
else:
    google_auth = LazyLoader("google_auth", globals(), "google.auth")
    google_auth_credentials = LazyLoader(
        "google_auth_credentials", globals(), "google.auth.credentials"
    )
    tts = LazyLoader("tts", globals(), "google.cloud.texttospeech_v1beta1")


set_basic_logging_config()
logger = logging.getLogger(__name__)
XML_TAGS = re.compile("<.*?>")


def _make_random_acronym(max_acronym_length: int) -> str:
    """Create a random acronym up to `max_acronym_length` in length."""
    chars = string.ascii_uppercase
    acronym = [random.choice(chars) for _ in range(random.randint(1, max_acronym_length))]
    acronym = "".join(acronym)
    return f'<say-as interpret-as="characters">{acronym}</say-as>'


def _gcp_synthesize_text(
    args: typing.Tuple[str, str],
    wavs_path: pathlib.Path,
    credentials: google_auth_credentials.Credentials,
    audio_config: tts.AudioConfig = tts.AudioConfig(
        audio_encoding=tts.AudioEncoding.LINEAR16,
        sample_rate_hertz=24000,
    ),
    voice: tts.VoiceSelectionParams = tts.VoiceSelectionParams(
        language_code="en-US", name="en-US-Wavenet-D"
    ),
):
    """Synthesizes speech from the input string of text.

    NOTE: "en-US-Wavenet-D" is the highest quality Google TTS voice based on MOS, learn more:
    https://cloud.google.com/blog/products/ai-machine-learning/cloud-text-to-speech-expands-its-number-of-voices-now-covering-33-languages-and-variants
    """
    ssml, file_name = args
    client = tts.TextToSpeechClient(credentials=credentials)
    input_text = tts.SynthesisInput(ssml=f"<speak>{ssml}</speak>")
    request = {"input": input_text, "voice": voice, "audio_config": audio_config}
    response = client.synthesize_speech(request=request)
    (wavs_path / f"{file_name}.wav").write_bytes(response.audio_content)  # type: ignore


def _make_words(
    max_words: typing.Optional[int], num_acronyms: int, max_acronym_length: int
) -> typing.List[str]:
    """Get a list of words with unambiguous pronunciations."""
    words = [
        word.lower()
        for word, pronunciations in load_cmudict_syl().items()
        if len(pronunciations) == 1
    ]
    logger.info(f"Found unambiguous {len(words)} words... and limiting to {max_words} words.")
    if max_words is not None:
        words = words[:max_words]
    acronyms = [_make_random_acronym(max_acronym_length) for _ in range(num_acronyms)]
    logger.info(f"Made {len(acronyms)} acronyms like: {acronyms[:10]}")
    words += acronyms
    return words


def main(
    root: pathlib.Path = run._config.DATA_PATH,
    num_parallel: int = 1,
    dataset_name: str = "gcp_pronunciation_dictionary",
    metadata_file_name: str = "metadata.csv",
    wavs_dir_name: str = "wavs",
    max_words: typing.Optional[int] = None,
    num_acronyms: int = 5000,
    max_acronym_length: int = 10,
):
    """
    Create a pronunciation dataset that closely follows the same format as conventional datasets
    like LJ Speech and M-AI Labs.
    """
    credentials = google_auth.default()[0]

    root = root / dataset_name
    logger.info(f"The dataset root directory will be `{root}`.")
    root.mkdir(exist_ok=False)

    words = _make_words(max_words, num_acronyms, max_acronym_length)

    file_names = [f"{i}_{re.sub(XML_TAGS, '', w)}".lower() for i, w in enumerate(words)]
    metadata = [f"{f}|{re.sub(XML_TAGS, '', w)}" for f, w in zip(file_names, words)]
    (root / metadata_file_name).write_text("\n".join(metadata))
    wavs_path = root / wavs_dir_name
    wavs_path.mkdir(exist_ok=False)
    logger.info(f"Synthesizing {len(words)} words....")
    with multiprocessing.pool.ThreadPool(num_parallel) as pool:
        partial_ = partial(_gcp_synthesize_text, wavs_path=wavs_path, credentials=credentials)
        list(tqdm(pool.imap(partial_, zip(words, file_names)), total=len(words)))


if __name__ == "__main__":
    app = typer.Typer(context_settings=dict(max_content_width=math.inf))
    typer.run(main)

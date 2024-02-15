import argparse
import os
import time
from functools import partial
from multiprocessing import Pool

import numpy as np
import pandas as pd
from tqdm import tqdm
from utils.mcd import (
    APIInput,
    AudioData,
    get_mcd,
    query_tts_api,
)
from respellings import MISPRONOUNCED
from generate._speaker_ids import MODEL_TO_SPEAKERS

SPEAKERS_IN_V9_V10_V11_V11_1 = {
    MODEL_TO_SPEAKERS["v9"] &
    MODEL_TO_SPEAKERS["v10"] &
    MODEL_TO_SPEAKERS["v11"] &
    MODEL_TO_SPEAKERS["v11-1"]
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-versions", "-m", required=True, help="Comma-separated models to evaluate"
    )
    parser.add_argument(
        "--df-path",
        "-df",
        required=True,
        help="Where to store the resulting dataframe",
    )
    parser.add_argument(
        "--clips-per-word",
        "-c",
        required=False,
        help="How many clips per word to generate",
        default=3,
        type=int
    )
    args = parser.parse_args()
    args.model_versions = args.model_versions.split(",")
    return args


def get_idx(iterable, item):
    """Wrapper for list.index() to make mapping easier"""
    return iterable.index(item)


def main(
    model_versions,
    df_path,
    speakers=SPEAKERS_IN_V9_V10_V11_V11_1,
    clips_per_word=3,
    debug=False
):
    words_and_speakers = [
        APIInput(
            text=word,
            respell_xml=respell[0],
            respell_colon=respell[1],
            speaker=speaker,
            speaker_id=speaker_id,
            model_version=model_version,
        )
        for word, respell in MISPRONOUNCED.items()
        for speaker, speaker_id in speakers
        for model_version in model_versions
    ]

    tasks = words_and_speakers * clips_per_word
    tasks = tasks[:10] if debug else tasks
    exceptions = []
    start = time.time()
    nproc = 5
    tts_word_arrays = []
    print("Querying API...")
    with Pool(nproc) as executor:
        with tqdm(total=len(tasks)) as pbar:
            for i in executor.imap(query_tts_api, tasks):
                if isinstance(i, AudioData):
                    tts_word_arrays.append(i)
                elif isinstance(i, str):
                    exceptions.append(i)
                pbar.update()

    print(f"Finished in {round(time.time() - start, 2)}s")
    if exceptions:
        print(f"Problem with {len(exceptions)} recordings")

    final_audio_data = []
    print("Calculating MCD...")
    with Pool(os.cpu_count()) as executor:
        with tqdm(total=len(tts_word_arrays)) as pbar:
            for i in executor.imap(get_mcd, tts_word_arrays):
                if isinstance(i, AudioData):
                    final_audio_data.append(i)
                pbar.update()

    print("Processing DataFrame...")
    df = pd.DataFrame(final_audio_data)
    np.set_printoptions(threshold=np.inf)
    words = [i for i in MISPRONOUNCED]

    for col in ["model_version", "text", "respell_text", "speaker"]:
        df[f"{col}"] = df[f"{col}"].astype("category")

    idx_getter = partial(get_idx, words)
    df.sort_values(
        by="text", axis=0, inplace=True, key=lambda col: col.map(idx_getter)
    )

    df[
        [
            "speaker",
            "text",
            "wav_data",
            "respell_text",
            "respell_wav_data",
            "mcd_dtw_value",
            "mcd_dtw_sl_value",
            "model_version",
        ]
    ].to_parquet(df_path)


if __name__ == "__main__":
    args = parse_args()
    main(
        model_versions=args.model_versions,
        df_path=args.df_path,
        clips_per_word=args.clips_per_word,
        debug=True
    )
9
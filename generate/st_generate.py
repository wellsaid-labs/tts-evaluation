"""Function to send requests to WSL's API. Saves audio and a .csv containing metadata to a zip file
USAGE:
$ python -m streamlit run generate/st_generate.py --runner.magicEnabled=false
"""

import logging
import random
import typing

import pandas as pd
import streamlit as st
from _speaker_ids import MODEL_TO_SPEAKERS
from _test_cases.report_card_test_cases import REPORT_CARD_TEST_CASES
from _test_cases.slurring import SLURRING
from _test_cases.v11_test_cases import V11_TEST_CASES
from dotenv import dotenv_values
from utils._streamlit import (
    WebPath,
    audio_to_web_path,
    make_temp_web_dir,
    st_download_files,
)
from generate.utils.api import APITransaction, query_wsl_api

logger = logging.getLogger(__name__)

test_case_options = V11_TEST_CASES
test_case_options["SLURRING"] = SLURRING
test_case_options["REPORT_CARD_TEST_CASES"] = REPORT_CARD_TEST_CASES
test_case_options = {
    k: v for k, v in sorted(test_case_options.items(), key=lambda x: x[0])
}

api_keys = dotenv_values(".env")


def process_task(task: APITransaction) -> typing.Tuple[dict, WebPath]:
    """Wrapper for query_wsl_api()
    Args:
        task (APITransaction): The input to the API
    Returns:
        Tuple containing an entry to audio metadata and the audio web path
    """
    audio_file_name = (
        f"{task.model_version}_{task.speaker}_{task.text[:25]}.wav"
    )
    task = query_wsl_api(task)
    audio_path = audio_to_web_path(task.wav_data, name=audio_file_name)
    metadata_entry = {
        "Speaker": "_".join(task.speaker.split("_")[0:2]),
        "Style": task.speaker.split("_")[-1],
        "Speaker_ID": task.speaker_id,
        "Script": task.text,
        "Vote": "",
        "Issues": "",
        "Session": "",
        "Dialect": "",
        "Audio": audio_file_name,
    }
    return metadata_entry, audio_path


def main():
    # Configure parameters and select appropriate headers for the API
    st.set_page_config(layout="wide")
    st.header("Use this app to generate audio from WSL's API")
    model_version = st.selectbox(
        label="Model Version. Updating this also updates available speakers.",
        options=MODEL_TO_SPEAKERS,
    )
    opts_form = st.form("Options")
    texts = []
    speaker_selection = opts_form.multiselect(
        label="Speakers",
        options=MODEL_TO_SPEAKERS[model_version],
        default=MODEL_TO_SPEAKERS[model_version],
    )
    test_case_selection = opts_form.selectbox(
        label="Select from predefined set of test cases",
        options=test_case_options,
        index=None,
    )
    single_sentence_mode = opts_form.checkbox(
        label="Split test cases into single sentences. Does not apply to custom test cases"
    )
    custom_test_case = opts_form.text_area(
        label="Custom Test Case - Overrides Test Cases dropdown above"
    )
    sentence_limit = opts_form.slider(
        label="Max number of sentences to generate.",
        min_value=1,
        max_value=500,
        value=250,
    )
    if test_case_selection and model_version:
        texts = V11_TEST_CASES[test_case_selection]
        if single_sentence_mode:
            texts = [
                sentence
                for i in texts
                for sentence in i.split(".")
                if len(sentence.strip()) > 1
            ]
    elif custom_test_case and model_version:
        texts = [custom_test_case]
    if opts_form.form_submit_button("Generate audio!"):
        # Generate audio. This could be sped up with multiprocessing, but the API is limited to
        # 2.5 requests per second
        total_tasks = len(texts) * len(speaker_selection)
        if total_tasks < sentence_limit:
            st.write(
                f"Total tasks: {total_tasks} greater than sentence limit: "
                f"{sentence_limit}. Generating {total_tasks} clips"
            )
        total_tasks = (
            sentence_limit if total_tasks > sentence_limit else total_tasks
        )
        tasks = [
            APITransaction(
                text=text,
                speaker_id=speaker_id,
                speaker=speaker,
                model_version=model_version,
            )
            for speaker, speaker_id in speaker_selection
            for text in texts
        ]
        tasks = random.sample(tasks, total_tasks)
        metadata, download_paths = [], []
        st.write(f"Total audio files: {len(tasks)}")
        pbar = st.progress(0, text="Generating...")
        current_task = 0
        for task in tasks:
            metadata_entry, audio_path = process_task(task)
            if metadata_entry and audio_path:
                metadata.append(metadata_entry)
                download_paths.append(audio_path)
            current_task += 1
            progress_pct = round(current_task / sentence_limit, 4)
            progress_txt = f"Generating... {progress_pct * 100}% complete"
            pbar.progress(progress_pct, text=progress_txt)

        # Create metadata csv and save zipped audio
        metadata_path = make_temp_web_dir() / "metadata.csv"
        metadata_path.write_text(
            pd.DataFrame(metadata).sort_values(by="Speaker").to_csv()
        )
        download_paths.append(metadata_path)
        fname = "custom-input" if custom_test_case else test_case_selection
        st_download_files(
            f"{model_version}-{fname}-audio-and-metadata.zip",
            "💾  DOWNLOAD  💾",
            download_paths,
        )


if __name__ == "__main__":
    main()

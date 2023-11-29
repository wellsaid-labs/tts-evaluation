"""Function to send requests to WSL's API. Saves audio and a .csv containing metadata to a zip file
USAGE:
$ PYTHONPATH=. streamlit run run/review/tts/generate_audio_from_api.py --runner.magicEnabled=false
"""
from dotenv import dotenv_values
import tempfile
import time
from dataclasses import dataclass
from typing import Tuple, Optional

import pandas as pd
import random
import requests
import soundfile as sf
import streamlit as st
from _test_cases.slurring import SLURRING
from _test_cases.v11_test_cases import V11_TEST_CASES

from run._config import configure
from run._streamlit import audio_to_web_path, make_temp_web_dir, st_download_files


@dataclass
class APIInput:
    text: str
    speaker_id: int
    speaker: str
    model_version: str
    headers: dict
    endpoint: str


test_case_options = V11_TEST_CASES
test_case_options["SLURRING"] = SLURRING
test_case_options = {k: v for k, v in sorted(test_case_options.items(), key=lambda x: x[0])}

api_keys = dotenv_values(".env")

speakers_in_v9_v10_v11 = {
    "Ramona_J_Narration": 4,
    "Alana_B_Narration": 3,
    "Sofia_H_Narration": 8,
    "Vanessa_N_Narration": 10,
    "Isabel_V_Narration": 11,
    "Jeremy_G_Narration": 13,
    "Nicole_L_Narration": 14,
    "Paige_L_Narration": 15,
    "Tobin_A_Narration": 16,
    "Tristan_F_Narration": 18,
    "Patrick_K_Narration": 19,
    "Joe_F_Narration": 27,
}


def query_tts_api(task: APIInput, retry_number: int = 0) -> requests.Response:
    """Post a request to WSL's API and return the result
    Args:
        task (APIInput): An APIInput object containing the necessary data to post a request.
        retry_attempt_number (int): The number of times this function will be allowed to error
            before returning a default value
    Returns:
        response (Response): The response from the post to WSL's API
    """
    max_retries = 2
    json_data = {
        "speaker_id": task.speaker_id,
        "text": task.text,
        "consumerId": "id",
        "consumerSource": "source",
    }
    response = requests.Response()
    if retry_number <= max_retries:
        response = requests.post(
            task.endpoint,
            headers=task.headers,
            json=json_data,
        )
        if response.status_code != 200:
            st.write(
                f"Received status code {response.status_code}. Attempt {retry_number}/{max_retries}"
            )
            time.sleep(3)
            retry_number += 1
            query_tts_api(task, retry_number)
        return response
    else:
        st.write("Max retries reached. Returning empty Response object")
        return response


def process_task(task: APIInput) -> Tuple[Optional[str], Optional[str]]:
    """Wrapper for query_tts_api()
    Args:
        task (APIInput): The input to the API
    Returns:
        Tuple containing an entry to this run's metadata and the audio web path
    """
    metadata_entry, audio_path = None, None
    audio_file_name = f"{task.model_version}_{task.speaker}_{task.text[:25]}.wav"
    resp = query_tts_api(task)
    if resp.status_code == 200:
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".wav") as fp:
            fp.write(resp.content)
            fp.seek(0)
            audio_array, sample_rate = sf.read(fp.name, dtype="float32")
        audio_path = audio_to_web_path(audio_array, name=audio_file_name)
        metadata_entry = {
            "Speaker": "".join(task.speaker.split("_")[0:1]),
            "Style": task.speaker.split("_")[-1],
            "Script": task.text,
            "Vote": "",
            "Issues": "",
            "Session": "",
            "Dialect": "",
            "Audio": audio_file_name,
        }
    else:
        msg = (
            f"Received status code {resp.status_code} while generating "
            f"{audio_file_name}. Full content: {resp.content}"
        )
        st.write(msg)
    return metadata_entry, audio_path


def main():
    # Configure parameters and select appropriate headers for the API
    configure(overwrite=True)
    st.header("Use this app to generate audio from WSL's API")
    opts_form = st.form("Options")
    texts = []
    endpoint = "https://staging.tts.wellsaidlabs.com/api/text_to_speech/stream"
    model_version = opts_form.text_input(label="Model Version")
    headers = {
        "X-Api-Key": api_keys["KONG_STAGING"],
        "Accept-Version": model_version,
        "Content-Type": "application/json",
    }
    test_case_selection = opts_form.selectbox(label="Test Cases", options=test_case_options)
    single_sentence_mode = opts_form.checkbox(label="Split test cases into single sentences")
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
                sentence for i in texts for sentence in i.split(".") if len(sentence.strip()) > 1
            ]
    if opts_form.form_submit_button("Generate audio!"):
        # Generate audio. This could be sped up with multiprocessing, but the API is limited to
        # 2.5 requests per second
        total_tasks = len(texts) * len(speakers_in_v9_v10_v11)
        if total_tasks < sentence_limit:
            st.write(
                f"Total tasks: {total_tasks} greater than sentence limit: "
                f"{sentence_limit}. Generating {total_tasks} clips"
            )
        total_tasks = sentence_limit if total_tasks > sentence_limit else total_tasks
        tasks = [
            APIInput(
                text=text,
                speaker_id=speaker_id,
                speaker=speaker,
                model_version=model_version,
                headers=headers,
                endpoint=endpoint,
            )
            for speaker, speaker_id in speakers_in_v9_v10_v11.items()
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
        metadata_path.write_text(pd.DataFrame(metadata).sort_values(by="Speaker").to_csv())
        download_paths.append(metadata_path)
        st_download_files(
            f"{model_version}-{test_case_selection}-audio-and-metadata.zip",
            "💾  DOWNLOAD  💾",
            download_paths,
        )


if __name__ == "__main__":
    main()

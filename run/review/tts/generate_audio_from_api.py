"""Function to send requests to WSL's API. Saves audio and a .csv containing metadata to a zip file
USAGE:
$ PYTHONPATH=. streamlit run run/review/tts/generate_audio_from_api.py --runner.magicEnabled=false
"""

import tempfile
import time

import pandas as pd
import requests
import soundfile as sf
import streamlit as st
from _test_cases.slurring import SLURRING
from _test_cases.v11_test_cases import V11_TEST_CASES

from run._config import configure
from run._streamlit import (
    audio_to_web_path,
    make_temp_web_dir,
    st_download_files,
)

test_case_options = V11_TEST_CASES
test_case_options["SLURRING"] = SLURRING
test_case_options = {
    k: v for k, v in sorted(test_case_options.items(), key=lambda x: x[0])
}

staging_headers = {
    "X-Api-Key": "d5637035-23d6-472d-9b44-87001cd337dc",
    "Accept-Version": "v11",
    "Content-Type": "application/json",
}
prod_headers = {
    "X-Api-Key": "c20c3edd-f60a-40fc-83f1-bcdf33b47b8c",
    "Accept-Version": "v10",
    "Content-Type": "application/json",
}
previous_headers = {
    "X-Api-Key": "c20c3edd-f60a-40fc-83f1-bcdf33b47b8c",
    "Accept-Version": "v9",
    "Content-Type": "application/json",
}

model_to_endpoints_and_headers = {
    "v9": ("https://api.wellsaidlabs.com/v1/tts/stream", previous_headers),
    "v10": ("https://api.wellsaidlabs.com/v1/tts/stream", prod_headers),
    "v11": (
        "https://api.wellsaidlabs.com/v1/staging/tts/stream",
        staging_headers,
    ),
}

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


def query_tts_api(speaker_id, text, style, headers, endpoint):
    """Send speaker_id and word to TTS API and convert the resulting file to
    np.ndarray. kwargs are used here to allow for easy multiprocessing
    Args:
    Returns:
        response (Response): The response from the post to WSL's API
    """
    json_data = {
        "speaker_id": speaker_id,
        "text": text,
        "consumerId": "id",
        "consumerSource": "source",
    }
    response = requests.post(
        endpoint,
        headers=headers,
        json=json_data,
    )
    return response


def main():
    # Configure parameters and select appropriate headers for the API
    configure(overwrite=True)
    with st.form("Options"):
        texts = []
        endpoint, headers = None, None
        model_version = st.selectbox(
            label="Model Version", options=["v9", "v10", "v11"]
        )
        if model_version:
            endpoint, headers = model_to_endpoints_and_headers[model_version]
        test_case_selection = st.selectbox(
            label="Test Cases", options=test_case_options
        )
        single_sentence_mode = st.selectbox(
            label="Split into single sententces", options=["Yes", "No"]
        )
        sentence_limit = st.slider(
            label="Max number of sentences to generate. 500 sentences results in a ~200mb zip file",
            min_value=1,
            max_value=500,
            value=250,
        )
        if test_case_selection and model_version:
            texts = V11_TEST_CASES[test_case_selection]
            if single_sentence_mode == "Yes":
                texts = [
                    sentence
                    for i in texts
                    for sentence in i.split(".")
                    if len(sentence.strip()) > 1
                ]
        if st.form_submit_button("Generate audio!"):
            # Generate audio. This could be sped up with multiprocessing, but the API is limited to
            # 2.5 requests per second
            total_tasks = len(texts) * len(speakers_in_v9_v10_v11)
            total_tasks = (
                sentence_limit if total_tasks > sentence_limit else total_tasks
            )
            tasks = [
                {"text": text, "speaker_id": speaker_id, "speaker": speaker}
                for text in texts
                for speaker, speaker_id in speakers_in_v9_v10_v11.items()
            ][:total_tasks]
            metadata, download_paths = [], []
            st.write(f"Total audio files: {total_tasks}")
            pbar = st.progress(0, text="Generating...")
            current_task = 0
            for task in tasks:
                speaker = task["speaker"]
                speaker_id = task["speaker_id"]
                text = task["text"]
                audio_file_name = f"{model_version}_{speaker}_{text[:15]}.wav"
                try:
                    resp = query_tts_api(
                        speaker_id=speaker_id,
                        text=text,
                        style=speaker,
                        endpoint=endpoint,
                        headers=headers,
                    )
                except requests.exceptions.ConnectionError:
                    st.write(
                        "Connection error encountered, trying once more in 5s"
                    )
                    time.sleep(5)
                    resp = query_tts_api(
                        speaker_id=speaker_id,
                        text=text,
                        style=speaker,
                        endpoint=endpoint,
                        headers=headers,
                    )
                if resp.status_code == 200:
                    with tempfile.NamedTemporaryFile(
                        mode="wb", suffix=".wav"
                    ) as fp:
                        fp.write(resp.content)
                        fp.seek(0)
                        audio_array, sample_rate = sf.read(
                            fp.name, dtype="float32"
                        )
                    path = audio_to_web_path(audio_array, name=audio_file_name)
                    download_paths.append(path)
                    file_data = {
                        "Speaker": "".join(speaker.split("_")[0:1]),
                        "Style": speaker.split("_")[-1],
                        "Script": text,
                        "Vote": [],
                        "Note": "",
                        "Session": "",
                        "Dialect": "",
                        "Audio": audio_file_name,
                    }
                    metadata.append(file_data)
                else:
                    msg = (
                        f"Received status code {resp.status_code} while generating "
                        f"{audio_file_name}. Full content: {resp.content}"
                    )
                    st.write(msg)
                current_task += 1
                progress_pct = round(current_task / total_tasks, 4)
                progress_txt = f"Generating... {progress_pct * 100}% complete"
                pbar.progress(progress_pct, text=progress_txt)

            # Create metadata csv and save zipped audio
            metadata_path = make_temp_web_dir() / "metadata.csv"
            metadata_path.write_text(pd.DataFrame(metadata).to_csv())
            download_paths.append(metadata_path)
            st_download_files(
                f"{model_version}-{test_case_selection}-audio-and-metadata.zip",
                "💾  DOWNLOAD  💾",
                download_paths,
            )


if __name__ == "__main__":
    main()

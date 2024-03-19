import os
import typing

import pandas as pd
import streamlit as st
from dotenv import dotenv_values
from lib.speaker_ids import MODEL_TO_SPEAKERS
from package_utils._streamlit import (
    WebPath,
    audio_to_web_path,
    make_temp_web_dir,
    st_download_files,
)
from package_utils.environment import ROOT_PATH
from _utils import APITransaction, query_wsl_api, DATASET_TYPES, DatasetConfig

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


def process_tasks(tasks: typing.List[APITransaction]):
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
        progress_pct = round(current_task / len(tasks), 4)
        progress_txt = f"Generating... {progress_pct * 100}% complete"
        pbar.progress(progress_pct, text=progress_txt)

    metadata_path = make_temp_web_dir() / "metadata.csv"
    metadata_path.write_text(
        pd.DataFrame(metadata).sort_values(by="Speaker").to_csv()
    )
    download_paths.append(metadata_path)

    return download_paths


def main():
    st.set_page_config(layout="wide")
    st.header("Generate Audio from WSL's API")
    api_transactions = []
    speaker_selection = []
    dataset_config = None
    test_cases_path = f"{ROOT_PATH}/generate/lib/test_case_csvs"
    test_case_options = [
        i for i in os.listdir(test_cases_path) if i.endswith("csv")
    ]

    with st.container(border=True):
        c1, c2 = st.columns(2)
        with c1:
            st.write("### Text Options")
            predefined_text = st.multiselect(
                "Predefined test cases", test_case_options
            )
            test_case_paths = [
                f"{test_cases_path}/{i}" for i in predefined_text
            ]
            custom_text = st.text_input("Custom Text - Optional")
            clips_per_text = st.number_input(
                "Clips Per Text", min_value=1, max_value=10, format="%d"
            )
            limit = st.number_input(
                label="Max number of sentences to generate. -1 means infinity.",
                min_value=-1,
                value=-1,
                format="%d",
            )

        with c2:
            st.write("### Storage Options")
            gcs_path = st.text_input("GCS Path", value="test_evaluations")
            dataset_type = st.selectbox(
                "Dataset type - Comparative or Absolute", options=DATASET_TYPES
            )

    with st.sidebar:
        st.write("### Model Options")
        model_versions = st.multiselect(
            label="Model Version. "
            "Updating this also updates available speakers.",
            options=MODEL_TO_SPEAKERS,
        )
        custom_model = st.text_input("Custom model version")
        custom_speaker_id = st.number_input(
            "Custom speaker id", min_value=0, format="%d"
        )
        custom_speaker_name = st.text_input("Custom speaker")

    ready_conditions = all(
        [model_versions or custom_model, predefined_text or custom_text]
    )
    if ready_conditions and not custom_model:
        model_speakers = MODEL_TO_SPEAKERS[model_versions[0]]
        for i in model_versions:
            model_speakers = model_speakers & MODEL_TO_SPEAKERS[i]
        model_speakers = pd.DataFrame(
            model_speakers, columns=["name", "speaker_id"]
        ).sort_values(by="name", ignore_index=True)
        model_speakers["selected"] = True
        st.write(f"{model_speakers.shape[0]} available speakers")
        speaker_selection = st.data_editor(
            data=model_speakers,
            use_container_width=True,
            hide_index=True,
        )
        speaker_selection = speaker_selection[
            speaker_selection.selected == True
        ][["name", "speaker_id"]].values.tolist()
        dataset_config = DatasetConfig(
            model_versions=model_versions,
            speakers=speaker_selection,
            predefined_texts=test_case_paths,
            custom_texts=[custom_text] if custom_text else None,
            gcs_path=gcs_path,
            clips_per_text=clips_per_text,
            dataset_type=dataset_type,
            task_limit=limit,
        )
        api_transactions = dataset_config.get_api_transactions()

    elif ready_conditions and custom_model:
        speaker_selection = [(custom_speaker_name, custom_speaker_id)]

        dataset_config = DatasetConfig(
            model_versions=[custom_model],
            speakers=speaker_selection,
            predefined_texts=test_case_paths,
            custom_texts=[custom_text] if custom_text else None,
            gcs_path=gcs_path,
            clips_per_text=clips_per_text,
            dataset_type=dataset_type,
            task_limit=limit,
        )
        api_transactions = dataset_config.get_api_transactions()

    st.write(f"### :rainbow[Ready to generate!]\n")
    gen_stats = pd.DataFrame(
        [
            {
                "Test Cases": len(dataset_config.combined_texts) if
                dataset_config else 0,
                "Speakers": len(speaker_selection),
                "Models": model_versions,
                "Total Clips": len(api_transactions),
            }
        ]
    )
    st.dataframe(gen_stats, hide_index=True, use_container_width=True)

    if st.button("Generate"):
        st.write(api_transactions)
        fname = "custom-input" if custom_text else predefined_text
        model_name = (
            f"{','.join(model_versions) if model_versions else custom_model}"
        )
        if len(api_transactions) < 200:
            download_paths = process_tasks(api_transactions)
            st_download_files(
                f"{model_name}-{fname}-audio-and-metadata.zip",
                "💾  DOWNLOAD  💾",
                download_paths,
            )
        else:
            config_path = f"{fname}-dataset-config.yaml"
            dataset_config.to_file("yaml", config_path)
            st.write(
                "Too many tasks for streamlit, use bulk_generate via "
                "this command:"
            )
            st.write(f"`python bulk_generate.py {config_path}`")


if __name__ == "__main__":
    main()

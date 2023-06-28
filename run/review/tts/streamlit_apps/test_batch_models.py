"""This script generates a zip file per model experiment and test case chosen which contains audio
files generated from that model as well as a `metadata.csv`, which contains informaiton relevant to
those audio files. The output of this can be given to `evaluate_local_audio.py`. This script was
written during v11 testing and therefore the upstream branch must be `v11_fine_tune` for it to work.

USAGE: $ PYTHONPATH=. streamlit run run/review/tts/streamlit_apps/test_batch_models.py --runner.magicEnabled=false
"""
import json
import os
import shutil
import subprocess

from lib.utils import mazel_tov

import streamlit as st
from run.review.tts.test_cases.long_scripts import LONG_SCRIPTS
from run.review.tts.test_cases.parity_test_cases import PARITY_TEST_CASES
from run.review.tts.test_cases.test_cases import TEST_CASES
from run.review.tts.test_cases.v11_test_cases import V11_TEST_CASES

all_test_cases = dict()
all_test_cases.update(TEST_CASES)
all_test_cases.update(V11_TEST_CASES)
all_test_cases.update(PARITY_TEST_CASES)
all_test_cases["LONG_SCRIPTS"] = LONG_SCRIPTS


def locate_models_in_gcs():
    with st.form(key="Download_Models"):
        gsutil = shutil.which("gsutil")
        gsutil_ls = f"{gsutil} ls {st.session_state.models_path_gcs}"
        try:
            results = subprocess.run(
                gsutil_ls, shell=True, check=True, capture_output=True, text=True
            )
        except subprocess.CalledProcessError:
            st.write(
                f"No results found at `{st.session_state.models_path_gcs}`, try a different URL"
            )
            return

        checkpoints = [i for i in results.stdout.split() if i.endswith(".pt")]
        dirs = [i for i in results.stdout.split() if i.endswith("/")]
        if checkpoints:
            st.write(f"Found the following checkpoints at `{st.session_state.models_path_gcs}`:")
            st.write([c.split("/")[-1] for c in checkpoints])
            st.multiselect(
                "Select checkpoints to download:",
                options=checkpoints,
                format_func=lambda x: x.split("/")[-1],
                key="models_to_download",
            )
            st.form_submit_button("💾 Download Models 💾")

        elif dirs:
            st.markdown(
                "Found directories. Enter one of these in the box above and click **Try Again**."
            )
            st.write(dirs)
            st.form_submit_button("Try again", on_click=update_gcs_path)


def update_gcs_path():
    st.session_state.models_path_gcs = st.session_state.models_path_gcs


def download_models():
    dest_path = (
        f'/Users/{os.environ["USER"]}/{st.session_state.models_to_download[0].split("/")[-2]}'
    )
    gsutil = shutil.which("gsutil")
    if not os.path.exists(dest_path):
        os.mkdir(dest_path)
    for idx, model in enumerate(st.session_state.models_to_download):
        with st.spinner(
            f"Downloading model {idx + 1}/{len(st.session_state.models_to_download)} to `{dest_path}`"
        ):
            cp_cmd = f"{gsutil} cp {model} {dest_path}/"
            subprocess.run(cp_cmd, shell=True, check=True)

        st.session_state.models_path_local = dest_path
    st.success("Models downloaded!")
    return


def get_model_files():
    model_files = [
        os.path.join(root, f)
        for root, _, files in os.walk(st.session_state.models_path_local)
        for f in files
    ]
    return model_files


def patch(model_path, test_cases):
    git = shutil.which("git")
    py = shutil.which("python")
    idx = model_path.split("__")[0].split("/")[-1]
    exp_id = model_path.split("__")[1]

    git_reset = f"{git} reset --hard"
    subprocess.run(git_reset, shell=True, capture_output=True, check=True)

    patch_cmd = f'{py} -m run.utils.comet patch {exp_id} --overwrite --include="*.py"'
    subprocess.run(patch_cmd, shell=True, check=True)
    generate_audio_cmd = f"{py} -m run.review.tts.util.generate_audio -i {idx} -s {model_path} -t {json.dumps(test_cases)}"
    subprocess.run(generate_audio_cmd, shell=True, check=True)
    subprocess.run(git_reset, shell=True, capture_output=True, check=True)


def init_state():
    if any(
        i not in st.session_state
        for i in [
            "models_path_local",
            "models_path_gcs",
        ]
    ):
        st.session_state["models_path_local"] = ""
        st.session_state["models_path_gcs"] = ""


def main():
    st.set_page_config(layout="wide")
    init_state()

    st.markdown("# Spectrogram Evaluations")
    st.markdown("Use this workbook to download audio from multiple spectrogram checkpoints")

    if "models_to_download" in st.session_state:
        download_models()
    elif "selected_models" not in st.session_state:
        if not st.session_state.models_path_local:
            st.text_input("GCS path", key="models_path_gcs")
        if not st.session_state.models_path_gcs:
            st.text_input("Local path", key="models_path_local")
        if st.session_state.models_path_gcs:
            locate_models_in_gcs()

    if st.session_state.models_path_local:
        with st.form("Select_Options"):
            st.multiselect("Select test cases", options=all_test_cases, key="selected_test_cases")
            model_options = get_model_files()
            st.multiselect(
                "Select the checkpoints to evaluate:",
                options=model_options,
                format_func=lambda x: x.split("/")[-1],
                key="selected_models",
            )

            st.form_submit_button("Continue")

    if (
        "FormSubmitter:Select_Options-Continue" in st.session_state
        and st.session_state["FormSubmitter:Select_Options-Continue"]
    ):
        for model in st.session_state.selected_models:
            with st.spinner(f"Generating audio with `{model.split('/')[-1]}`"):
                patch(model, str(st.session_state.selected_test_cases))
        st.success("Audio generated!", icon=mazel_tov())


if __name__ == "__main__":
    main()

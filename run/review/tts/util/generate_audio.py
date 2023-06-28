import argparse
import typing
import random
import pandas as pd
import zipfile
import os
import tempfile

from functools import partial
from pathlib import Path

import run
from lib.audio import write_audio
from lib.environment import load
from lib.text.utils import XMLType
from run._models.spectrogram_model import SpectrogramModel
from run._tts import griffin_lim_tts
from run.data._loader import Session
from run.deploy.worker import _MARKETPLACE

from run.review.tts.test_cases.long_scripts import LONG_SCRIPTS
from run.review.tts.test_cases.parity_test_cases import PARITY_TEST_CASES
from run.review.tts.util.speaker_ids import SPEAKER_IDS
from run.review.tts.test_cases.test_cases import TEST_CASES
from run.review.tts.test_cases.v11_test_cases import V11_TEST_CASES

all_test_cases = dict()
all_test_cases.update(TEST_CASES)
all_test_cases.update(V11_TEST_CASES)
all_test_cases.update(PARITY_TEST_CASES)
all_test_cases["LONG_SCRIPTS"] = LONG_SCRIPTS


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument("-s", "--spectrogram", help="Spectrogram model to load")
    args.add_argument("-t", "--test-cases", help="Dict of test cases to generate")
    args.add_argument("-i", "--index", help="Index to distinguish the current model")
    args = args.parse_args()
    test_case = ""
    test_cases = []
    for i in args.test_cases:
        if i.isalnum() or i == "_":
            test_case += i
        else:
            test_cases.append(test_case)
            test_case = ""
    args.test_cases = [t for t in test_cases if t]
    return args


def get_spectrogram_export(model_path):
    run._config.configure(overwrite=True)
    checkpoint_path = Path(model_path)
    spec_ckpt = typing.cast(run.train.spectrogram_model._worker.Checkpoint, load(checkpoint_path))
    spec_export = spec_ckpt.export()
    return spec_export


def generate_test_cases(
    spec_export: SpectrogramModel, test_cases: typing.List[str], seed: int = 123
):
    session_vocab = set(spec_export.session_embed.vocab)
    assert session_vocab
    mkt_names_and_labels = [(v[0], v[1]) for k, v in _MARKETPLACE.items() if k in SPEAKER_IDS]
    assert mkt_names_and_labels
    spk_sessions = [
        s
        for s in session_vocab
        if isinstance(s, Session) and (s.spkr, s.label) in mkt_names_and_labels
    ]
    assert spk_sessions
    for sp_s in spk_sessions:
        if sp_s not in session_vocab:
            print(f"Speaker missing: {sp_s}")
    for case in test_cases:
        sesh = random.choice(spk_sessions)
        yield (sesh, case, griffin_lim_tts(spec_export, XMLType(case), sesh))


def main():
    args = parse_args()
    test_cases = {k: all_test_cases[k] for k in args.test_cases}
    spec_export = get_spectrogram_export(args.spectrogram)
    test_cases = sorted(
        {k: partial(generate_test_cases, test_cases=v) for k, v in test_cases.items()}.items()
    )
    for test_case in test_cases:
        case_name = test_case[0]
        audio = test_case[1](spec_export)
        audios, audio_paths, rows = [], [], []
        audios.extend(audio)

        with tempfile.TemporaryDirectory() as tmp_dir:
            for i, (sesh, script, wave) in enumerate(audios):
                audio_path = os.path.join(tmp_dir, f"{case_name}_{i}.wav")
                audio_paths.append(audio_path)
                row = {
                    "Speaker": sesh.spkr.label,
                    "Style": sesh.spkr.style.value,
                    "Script": script,
                    "Vote": "",
                    "Note": "",
                    "Session": sesh.label,
                    "Dialect": sesh.spkr.dialect.value[1],
                    "Audio": f"{case_name}_{i}.wav",
                }
                write_audio(audio_path, wave, sample_rate=24000)  # type: ignore
                rows.append(row)
            df = pd.DataFrame(rows)
            metadata_path = os.path.join(tmp_dir, "metadata.csv")
            df.to_csv(metadata_path)
            with zipfile.ZipFile(f"/Users/jordan/Downloads/{args.index}_{case_name}.zip", "w") as z:
                audio_zip_paths = [i.split("/")[-1] for i in audio_paths]
                metadata_zip_path = metadata_path.split("/")[-1]
                for audio_path, audio_zip_path in zip(audio_paths, audio_zip_paths):
                    z.write(audio_path, audio_zip_path)
                z.write(metadata_path, metadata_zip_path)


if __name__ == "__main__":
    main()

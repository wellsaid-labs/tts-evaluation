""" Tool for sampling passages for comparison with deployment.

TODO: Should `run._config.DEV_SPEAKERS` be a configured parameter?

Usage:

    $ python -m run.evaluate.sample_passages
"""
import io
import logging
import pathlib
import random
import zipfile

import pandas
import typer

import lib
import run

lib.environment.set_basic_logging_config()
logger = logging.getLogger(__name__)
app = typer.Typer()


@app.command()
def main(
    path: pathlib.Path = run._config.TEMP_PATH / "sample_of_passages.zip",
    csv_name: str = "scripts.csv",
    file_name_column_name: str = "File Name",
    script_column_name: str = "Script",
    num_samples: int = typer.Option(5, help="Write this many samples per speaker."),
    overwrite: bool = typer.Option(False, help="Overwrite PATH if it exists."),
    use_dev_dataset: bool = typer.Option(
        True, help="Only include passages from the development dataset."
    ),
    only_dev_speakers: bool = typer.Option(True, help="Only sample from the development speakers."),
    filter_on_session: bool = typer.Option(
        True, help="Only include passages with from the preferred recording session."
    ),
    debug: bool = False,
):
    """Sample passages from the dataset for comparison with deployment."""
    if path.exists() and not overwrite:
        typer.echo(f"The file {path} already exists.")
        raise typer.Exit()
    if not path.parent.exists():
        typer.echo(f"The directory {path.parent} doesn't exist.")
        raise typer.Exit()

    run._config.configure()

    _datasets = {k: v for k, v in list(run._config.DATASETS.items())[:1]}
    dataset = run._utils.get_dataset(**({"datasets": _datasets} if debug else {}))
    if use_dev_dataset:
        _, dataset = run._utils.split_dataset(dataset)

    sessions = {spk: sesh for spk, sesh in run.deploy.worker.SPEAKER_ID_TO_SPEAKER.values()}
    data = []
    logger.info(f"Writing Zip file to {path}")
    with zipfile.ZipFile(path, "w") as file_:
        for speaker, passages in dataset.items():
            if speaker not in run._config.DEV_SPEAKERS and only_dev_speakers:
                continue
            if filter_on_session:
                passages = [p for p in passages if p.session == sessions[speaker]]
            sample = random.sample(passages, min(num_samples, len(passages)))
            for i, passage in enumerate(sample):
                sesh = str(passage.session).replace("/", "__")
                file_name = f"spk={speaker.label},sesh={sesh},id={i}.wav"
                data.append({file_name_column_name: file_name, script_column_name: passage.script})
                mock_file_ = io.BytesIO()
                lib.audio.write_audio(mock_file_, passage[:].audio())
                file_.writestr(file_name, mock_file_.read())
        file_.writestr(csv_name, pandas.DataFrame(data).to_csv(index=False))
    logger.info(f"Finished! {lib.utils.mazel_tov()}")


if __name__ == "__main__":
    app()

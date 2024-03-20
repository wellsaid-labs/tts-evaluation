import sys
import time
from multiprocessing import Pool
from pprint import pformat

from generate._utils import (
    APITransaction,
    query_wsl_api,
    DatasetConfig,
    AudioDataset,
)
from package_utils.environment import logger
from tqdm import tqdm


def main(
    audio_dataset,
    model_versions,
    speakers,
    texts,
    clips_per_text,
    debug=False,
):
    tasks = (
        APITransaction(
            text=text,
            speaker=speaker,
            speaker_id=speaker_id,
            model_version=model_version,
        )
        for text in texts
        for speaker, speaker_id in speakers
        for model_version in model_versions
        for _ in range(clips_per_text)
    )
    tasks = list(tasks)[:10] if debug else tasks
    exceptions = []
    start = time.time()
    nproc = 5
    logger.info("Querying API...")
    with Pool(nproc) as executor:
        with tqdm(total=len(tasks)) as pbar:
            for i in executor.imap(query_wsl_api, tasks):
                if isinstance(i, APITransaction):
                    audio_dataset.audio.append(i)
                else:
                    exceptions.append(i)
                pbar.update()

        logger.info(f"Finished in {round(time.time() - start, 2)}s")
    if exceptions:
        logger.info(f"Problem with {len(exceptions)} recordings")


if __name__ == "__main__":
    run_config = DatasetConfig.from_file(sys.argv[1])
    logger.info(f"Dataset Configuration: \n{pformat(run_config.as_dict())}")
    audio_dataset = AudioDataset(config=run_config, audio=list())

    main(
        audio_dataset=audio_dataset,
        debug=True,
        model_versions=run_config.model_versions,
        speakers=run_config.speakers,
        texts=run_config.combined_texts,
        clips_per_text=run_config.clips_per_text,
    )
    audio_dataset.upload_blob_from_memory()

import logging
import time
import sys
sys.path.append(".")
from multiprocessing import Pool

from tqdm import tqdm
from generate._utils.api import APITransaction, query_wsl_api
from generate._utils.structures import AudioDataset, DatasetConfig
logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)


def main(
    audio_dataset,
    model_versions,
    gcs_path,
    speakers,
    texts,
    dataset_type,
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
    # args = parse_args()
    run_config = DatasetConfig.from_json("/Users/jordan/Workspaces/tts-evaluation/generate/configs/test_config.json")
    audio_dataset = AudioDataset(config=run_config, audio=list())

    main(
        audio_dataset=audio_dataset,
        debug=True,
        **run_config.as_dict(),
    )

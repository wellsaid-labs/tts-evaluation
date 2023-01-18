""" Python application for analyzing fully processed training data.

TODO: Add tests to `test__main__.py` with additional coverage of the fully processed training data.

Usage:
    $ python -m run.review.dataset_processing.fully_processed_training_data
"""
import logging

import config as cf

import lib
import run
from run._utils import get_datasets
from run.train.spectrogram_model._worker import _get_data_generator, _get_data_processors

lib.environment.set_basic_logging_config(reset=True)
logger = logging.getLogger(__name__)


def main():
    run._config.configure(overwrite=True)
    train_dataset, dev_dataset = get_datasets(False)
    cf.add(run._config.make_spectrogram_model_train_config(train_dataset, dev_dataset, False))
    train_gen, dev_gen = cf.partial(_get_data_generator)(train_dataset, dev_dataset)
    train_process, _ = cf.partial(_get_data_processors)(train_gen, dev_gen, 0)
    batch_idx = 0
    batch = train_process[batch_idx]
    processed = batch.processed
    for seq_idx in range(len(batch)):
        print(f"XML: '{batch.xmls[seq_idx]}'")
        print(
            "Sequence Metadata:",
            processed.seq_meta[0][seq_idx],
            processed.seq_meta[1][seq_idx],
            processed.seq_meta[2][seq_idx],
            processed.seq_meta[3][seq_idx],
            processed.seq_meta[4][seq_idx],
        )
        iter_ = zip(
            processed.tokens[seq_idx],
            processed.token_meta[0][seq_idx],
            processed.token_meta[1][seq_idx],
        )
        for idx, (token, casing, cntxt) in enumerate(iter_):
            print(f"Token: '{repr(token)}'")
            print("Token Metadata:", casing, cntxt)
            annos = (
                "loudness_anno_embed",
                "loudness_anno_mask",
                "sesh_loudness_embed",
                "tempo_anno_embed",
                "tempo_anno_mask",
                "sesh_tempo_embed",
            )
            for anno in annos:
                print(anno, processed.get_token_vec(anno)[idx])
        print("=" * 10)


if __name__ == "__main__":
    main()

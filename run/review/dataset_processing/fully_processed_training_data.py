""" Python application for analyzing fully processed training data.

TODO: Add tests to `test__main__.py` with additional coverage of the fully processed training data.

Usage:
    $ python -m run.review.dataset_processing.fully_processed_training_data
"""
import logging

import config as cf
import torch

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
            processed.seq_metadata[0][seq_idx],
            processed.seq_metadata[1][seq_idx],
            processed.seq_metadata[2][seq_idx],
            processed.seq_metadata[3][seq_idx],
            processed.seq_metadata[4][seq_idx],
        )
        iter_ = zip(
            processed.tokens[seq_idx],
            processed.token_metadata[0][seq_idx],
            processed.token_metadata[1][seq_idx],
        )
        for idx, (token, casing, cntxt) in enumerate(iter_):
            print(f"Token: '{repr(token)}'")
            print("Token Metadata:", casing, cntxt)
            assert isinstance(processed.token_embeddings, torch.Tensor)
            print("Annotations:", processed.anno_embeddings)
        print("=" * 10)


if __name__ == "__main__":
    main()

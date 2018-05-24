""" Generate relevant training files for the Wavenet model.
"""

import matplotlib
matplotlib.use('Agg')

import argparse
import logging
import os
import random

from tqdm import tqdm

import torch
import tensorflow as tf

from src.bin.feature_model._utils import DataIterator
from src.bin.feature_model._utils import load_checkpoint
from src.bin.feature_model._utils import load_data
from src.bin.feature_model._utils import set_hparams as set_feature_model_auxiliary_hparams
from src.hparams import set_hparams
from src.utils import get_total_parameters
from src.utils import torch_save

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main(checkpoint, destination='data/signal_dataset.pt', use_multiprocessing=True,
         batch_size=64):  # pragma: no cover
    """ Main module used to generate dataset for training a signal model.

    Args:
        checkpoint (str): Checkpoint to load used to generate.
        destination (str, optional): Directory to save generated files.
        use_multiprocessing (bool, optional): If `True`, use multiple processes to preprocess data.
        batch_size (int, optional): Batch size used to generate.
    """
    set_hparams()
    set_feature_model_auxiliary_hparams()

    if not os.path.isdir(destination):
        os.makedirs(destination)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    checkpoint = load_checkpoint(checkpoint, device)
    text_encoder = None if checkpoint is None else checkpoint['text_encoder']
    model = checkpoint['model'].to(checkpoint.device)
    train, dev, text_encoder = load_data(
        text_encoder=text_encoder, load_signal=True, use_multiprocessing=use_multiprocessing)

    logger.info('Device: %s', device)
    logger.info('Number of Train Rows: %d', len(train))
    logger.info('Number of Dev Rows: %d', len(dev))
    logger.info('Vocab Size: %d', text_encoder.vocab_size)
    logger.info('Batch Size: %d', batch_size)
    logger.info('Total Parameters: %d', get_total_parameters(model))
    logger.info('Model:\n%s' % model)

    torch.set_grad_enabled(False)
    model.train(mode=False)

    return_ = []

    for dataset in [train, dev]:

        processed = []
        data_iterator = tqdm(DataIterator(device, dataset, batch_size, load_signal=True))
        for i, (gold_texts, gold_text_lengths, gold_frames, gold_frame_lengths, _,
                gold_quantized_signals) in enumerate(data_iterator):

            _, batch_predicted_frames, _, _ = model(gold_texts, ground_truth_frames=gold_frames)
            # [num_frames, batch_size, frame_channels] → [batch_size, num_tokens, frame_channels]
            batch_predicted_frames = batch_predicted_frames.transpose_(0, 1).cpu()
            batch_size = batch_predicted_frames.shape[0]

            # Save predictions
            for j in range(batch_size):
                # [batch_size, num_tokens, frame_channels] → [num_tokens, frame_channels]
                predicted_frames = batch_predicted_frames[j][:gold_frame_lengths[j]]
                # [batch_size, signal_length] → [signal_length]
                quantized_signal = gold_quantized_signals[j].cpu()
                assert quantized_signal.shape[0] % predicted_frames.shape[0] == 0
                processed.append({
                    'log_mel_spectrogram': predicted_frames,  # [num_tokens, frame_channels]
                    'quantized_signal': quantized_signal,  # [signal_length]
                })

        # Shuffle to not be affected by ordering
        random.shuffle(processed)
        return_.append(processed)

    torch_save(destination, tuple(return_))


if __name__ == '__main__':  # pragma: no cover
    tf.enable_eager_execution()
    # LEARN MORE:
    # https://stackoverflow.com/questions/42270739/how-do-i-resolve-these-tensorflow-warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--checkpoint", type=str, default=None, help="load a checkpoint", required=True)
    parser.add_argument(
        "-nm",
        "--no_multiprocessing",
        default=False,
        action='store_true',
        help="Sometimes multiprocessing breaks due to various reasons, this bool lets you turn " +
        "off multiprocessing.")
    args = parser.parse_args()
    main(checkpoint=args.checkpoint, use_multiprocessing=not args.no_multiprocessing)

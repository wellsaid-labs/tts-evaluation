""" Generate relevant training files for the Wavenet model.
"""
import argparse
import logging
import os

from tqdm import tqdm

import torch
import numpy as np

from src.bin.feature_model._utils import DataIterator
from src.bin.feature_model._utils import load_checkpoint
from src.bin.feature_model._utils import load_data
from src.bin.feature_model._utils import set_hparams as set_feature_model_auxiliary_hparams
from src.hparams import set_hparams
from src.utils import get_total_parameters

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main(checkpoint,
         destination_train='data/signal_dataset/train',
         destination_dev='data/signal_dataset/dev',
         use_multiprocessing=True,
         max_batch_size=96):  # pragma: no cover
    """ Main module used to generate dataset for training a signal model.

    Args:
        checkpoint (str): Checkpoint to load used to generate.
        destination_train (str, optional): Directory to save generated files to be used for
            training.
        destination_dev (str, optional): Directory to save generated files to be used for
            development.
        use_multiprocessing (bool, optional): If `True`, use multiple processes to preprocess data.
        max_batch_size (int, optional): Maximum batch size predicted at a time.
    """
    set_hparams()
    set_feature_model_auxiliary_hparams()

    if not os.path.isdir(destination_train):
        os.makedirs(destination_train)

    if not os.path.isdir(destination_dev):
        os.makedirs(destination_dev)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    checkpoint = load_checkpoint(checkpoint, device)
    text_encoder = None if checkpoint is None else checkpoint['text_encoder']
    model = checkpoint['model'].to(device)
    train, dev, text_encoder = load_data(
        text_encoder=text_encoder, load_signal=True, use_multiprocessing=use_multiprocessing)

    logger.info('Device: %s', device)
    logger.info('Number of Train Rows: %d', len(train))
    logger.info('Number of Dev Rows: %d', len(dev))
    logger.info('Vocab Size: %d', text_encoder.vocab_size)
    logger.info('Maximum batch Size: %d', max_batch_size)
    logger.info('Total Parameters: %d', get_total_parameters(model))
    logger.info('Model:\n%s' % model)

    torch.set_grad_enabled(False)
    model.train(mode=False)

    for dataset, destination in [(train, destination_train), (dev, destination_dev)]:
        data_iterator = tqdm(DataIterator(device, dataset, max_batch_size, load_signal=True))
        for i, (gold_texts, gold_text_lengths, gold_frames, gold_frame_lengths, _,
                gold_quantized_signals) in enumerate(data_iterator):

            _, batch_predicted_frames, _, _ = model(gold_texts, ground_truth_frames=gold_frames)
            # [num_frames, batch_size, frame_channels] → [batch_size, num_frames, frame_channels]
            batch_predicted_frames = batch_predicted_frames.transpose(0, 1).cpu().numpy().astype(
                np.float32)

            batch_size = batch_predicted_frames.shape[0]

            # Save predictions
            for j in range(batch_size):
                # [batch_size, num_frames, frame_channels] → [num_frames, frame_channels]
                predicted_frames = batch_predicted_frames[j][:gold_frame_lengths[j]]
                # [batch_size, signal_length] → [signal_length]
                quantized_signal = gold_quantized_signals[j].cpu().numpy().astype(np.int16)
                assert quantized_signal.shape[0] % predicted_frames.shape[0] == 0

                # NOTE: ``numpy .npy (no pickle)`` is about 15 times faster than numpy with pickle
                # ``torch.save`` does not include none-pickle options; therefore, we use numpy
                # Performance is important here because we are saving 10,000+ files.
                np.save(
                    os.path.join(destination, 'log_mel_spectrogram_%d_%d.npy' % (i, j)),
                    predicted_frames,
                    allow_pickle=False)
                np.save(
                    os.path.join(destination, 'quantized_signal_%d_%d.npy' % (i, j)),
                    quantized_signal,
                    allow_pickle=False)


if __name__ == '__main__':  # pragma: no cover
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

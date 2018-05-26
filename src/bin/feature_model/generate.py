""" Generate relevant training files for the Wavenet model.
"""

import matplotlib
matplotlib.use('Agg')

import argparse
import logging
import os

from tqdm import tqdm

import torch
import tensorflow as tf
import numpy as np

from src.bin.feature_model._utils import DataIterator
from src.bin.feature_model._utils import load_checkpoint
from src.bin.feature_model._utils import load_data
from src.hparams import set_hparams
from src.utils import get_total_parameters

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main(checkpoint, destination='data/vocoder/', use_multiprocessing=True,
         batch_size=128):  # pragma: no cover
    """ Main module.

    TODO: Rewrite for my own Wavenet implementation.

    Args:
        checkpoint (str): Checkpoint to load used to generate.
        destination (str, optional): Directory to save generated files.
        use_multiprocessing (bool, optional): If `True`, use multiple processes to preprocess data.
    """
    set_hparams()

    if not os.path.isdir(destination):
        os.makedirs(destination)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    checkpoint = load_checkpoint(checkpoint, device)
    text_encoder = None if checkpoint is None else checkpoint['text_encoder']
    model = checkpoint['model']
    train, dev, text_encoder = load_data(
        text_encoder=text_encoder, load_signal=True, use_multiprocessing=use_multiprocessing)
    data = train + dev

    logger.info('Device: %s', device)
    logger.info('Number of Rows: %d', len(data))
    logger.info('Vocab Size: %d', text_encoder.vocab_size)
    logger.info('Batch Size: %d', batch_size)
    logger.info('Total Parameters: %d', get_total_parameters(model))
    logger.info('Model:\n%s' % model)

    torch.set_grad_enabled(False)
    model.train(mode=False)

    data_iterator = tqdm(DataIterator(device, data, batch_size, train=False, load_signal=True))
    for i, (gold_texts, gold_text_lengths, gold_frames, gold_frame_lengths, _,
            gold_signals) in enumerate(data_iterator):
        _, batch_predicted_frames, _, _ = model(gold_texts, ground_truth_frames=gold_frames)
        # [num_frames, batch_size, frame_channels] → [batch_size, num_tokens, frame_channels]
        batch_predicted_frames = batch_predicted_frames.transpose(0, 1).cpu().numpy()

        # Save predictions
        for j in range(batch_predicted_frames.shape[0]):
            predicted_frames = batch_predicted_frames[j][:gold_frame_lengths[j]]  # Cutoff padding
            signal = gold_signals[j].cpu().numpy()
            assert signal.shape[0] % predicted_frames.shape[0] == 0

            audio_filename = os.path.abspath(
                os.path.join(destination, 'ljspeech-audio-%d-%d.npy' % (i, j)))
            mel_filename = os.path.abspath(
                os.path.join(destination, 'ljspeech-mel-%d-%d.npy' % (i, j)))
            np.save(audio_filename, signal.astype(np.int16), allow_pickle=False)
            np.save(mel_filename, predicted_frames.astype(np.float32), allow_pickle=False)


if __name__ == '__main__':  # pragma: no cover
    tf.enable_eager_execution()
    # LEARN MORE:
    # https://stackoverflow.com/questions/42270739/how-do-i-resolve-these-tensorflow-warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--checkpoint", type=str, default=None, help="load a checkpoint")
    parser.add_argument(
        "-nm",
        "--no_multiprocessing",
        default=False,
        action='store_true',
        help="Sometimes multiprocessing breaks due to various reasons, this bool lets you turn " +
        "off multiprocessing.")
    args = parser.parse_args()
    main(checkpoint=args.checkpoint, use_multiprocessing=not args.no_multiprocessing)

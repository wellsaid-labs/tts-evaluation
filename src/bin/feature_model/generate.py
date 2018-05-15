""" Generate relevant files for the ``r9y9/wavenet_vocoder``.
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
from src.utils import get_total_parameters
from src.hparams import set_hparams

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main(checkpoint, destination='data/vocoder/'):  # pragma: no cover
    """ Main module.

    Notes:
        * ``r9y9/wavenet_vocoder`` does it's own test and train split; therefore, we do not have
            control over that split. It may be difficult to find text files neither model has
            trained on.

    Args:
        checkpoint (str): Checkpoint to load used to generate.
        destination (str, optional): Directory to save generated files.
    """
    set_hparams()

    if not os.path.isdir(destination):
        os.makedirs(destination)

    device = torch.cuda.current_device() if torch.cuda.is_available() else -1
    checkpoint = load_checkpoint(checkpoint, device)
    text_encoder = None if checkpoint is None else checkpoint['text_encoder']
    model = checkpoint['model']
    train, dev, text_encoder = load_data(text_encoder=text_encoder, load_signal=True)
    data = train + dev
    metadata = open(os.path.join(destination, 'train.txt'), 'w+')
    batch_size = 128

    logger.info('Device: %d', device)
    logger.info('Number of Rows: %d', len(data))
    logger.info('Vocab Size: %d', text_encoder.vocab_size)
    logger.info('Batch Size: %d', batch_size)
    logger.info('Total Parameters: %d', get_total_parameters(model))
    logger.info('Model:\n%s' % model)

    torch.set_grad_enabled(False)
    model.train(mode=False)

    data_iterator = tqdm(DataIterator(device, data, batch_size, train=False, load_signal=True))
    for i, (gold_texts, gold_frames, gold_frame_lengths, _,
            gold_signals) in enumerate(data_iterator):
        _, batch_predicted_frames, _, _ = model(gold_texts, ground_truth_frames=gold_frames)
        # [num_frames, batch_size, frame_channels] → [batch_size, num_tokens, frame_channels]
        batch_predicted_frames = batch_predicted_frames.transpose(0, 1).cpu().numpy()

        # Save predictions
        for j in range(batch_predicted_frames.shape[0]):
            length = gold_frame_lengths[j]
            predicted_frames = batch_predicted_frames[j][:length]  # Cutoff padding
            signal = gold_signals[j].cpu().numpy()
            text = text_encoder.decode(gold_texts[j])
            assert signal.shape[0] % predicted_frames.shape[0] == 0

            audio_filename = os.path.abspath(
                os.path.join(destination, 'ljspeech-audio-%d-%d.npy' % (i, j)))
            mel_filename = os.path.abspath(
                os.path.join(destination, 'ljspeech-mel-%d-%d.npy' % (i, j)))
            np.save(audio_filename, signal.astype(np.int16), allow_pickle=False)
            np.save(mel_filename, predicted_frames.astype(np.float32), allow_pickle=False)
            metadata.write('|'.join([audio_filename, mel_filename, str(len(signal)), text]) + '\n')
    metadata.close()


if __name__ == '__main__':  # pragma: no cover
    tf.enable_eager_execution()
    # LEARN MORE:
    # https://stackoverflow.com/questions/42270739/how-do-i-resolve-these-tensorflow-warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--checkpoint", type=str, default=None, help="load a checkpoint")
    args = parser.parse_args()
    main(checkpoint=args.checkpoint)

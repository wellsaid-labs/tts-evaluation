import matplotlib
matplotlib.use('Agg')

import argparse
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import torch
import tensorflow as tf

tf.enable_eager_execution()

from src.utils.experiment_context_manager import load
from src.preprocess import plot_spectrogram
from src.preprocess import log_mel_spectrogram_to_wav
from src.utils import get_total_parameters
from src.utils import get_root_path
from src.utils import plot_attention
from src.hparams import set_hparams

# TODO: Generate ground truth aligned mel spectrogram predications for training vocoder


def sample_spectrogram(batch, directory, name):
    spectrogram = batch.transpose_(0, 1)[0].cpu().numpy()
    name = os.path.join(directory, name)
    plot_spectrogram(spectrogram, name + '.png')
    with tf.device('/cpu'):
        log_mel_spectrogram_to_wav(spectrogram, name + '.wav')


def sample_attention(batch, directory, filename):
    """ Sample an alignment from a batch and save a visualization.

    Args:
        batch (torch.FloatTensor [num_frames, batch_size, num_tokens]): Batch of alignments.
        filename (str): Filename to use for sample without an extension
    """
    filename = os.path.join(directory, filename)
    _, batch_size, _ = batch.shape
    alignment = batch.transpose_(0, 1)[0].cpu().numpy()
    plot_attention(alignment, filename + '.png')


def main():
    """ Main module if this file is invoked directly """
    set_hparams()

    # Load checkpoint
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--checkpoint", type=str, help="load a checkpoint")
    args = parser.parse_args()

    checkpoint_path = os.path.join(get_root_path(), args.checkpoint)
    checkpoint = load(os.path.join(get_root_path(), args.checkpoint))
    directory = os.path.dirname(checkpoint_path)
    logger.info('Loaded checkpoint: %s (%d step)' % (args.checkpoint, checkpoint['step']))
    text_encoder = checkpoint['text_encoder']

    model = checkpoint['model']
    model.apply(lambda m: m.flatten_parameters() if hasattr(m, 'flatten_parameters') else None)
    torch.set_grad_enabled(False)
    model.train(mode=False)

    logger.info('Total Parameters: %d', get_total_parameters(model))
    logger.info('Model:\n%s' % model)

    while True:
        text = input('Text: ').strip()
        logger.info('Got text: %s', text)
        text = text_encoder.encode(text).unsqueeze(0)
        frames, frames_with_residual, stop_tokens, alignments = model(text, max_recursion=10000)
        logger.info('Predicted Mel-spectrogram')
        sample_spectrogram(frames_with_residual, directory, 'generated_predicted_post_spectrogram')
        sample_spectrogram(frames, directory, 'generated_predicted_pre_spectrogram')
        sample_attention(alignments, directory, 'generated_attention')


if __name__ == '__main__':
    main()

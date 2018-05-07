import argparse
import logging
import os

import torch
import tensorflow as tf

tf.enable_eager_execution()

from src.experiment_context_manager import load
from src.spectrogram import plot_spectrogram
from src.spectrogram import log_mel_spectrogram_to_wav
from src.utils import get_total_parameters
from src.utils import get_root_path

logger = logging.getLogger(__name__)


def sample_spectrogram(batch, directory, name):
    spectrogram = batch.transpose_(0, 1)[0].cpu().numpy()
    name = os.path.join(directory, name)
    plot_spectrogram(spectrogram, name + '.png')
    with tf.device('/cpu'):
        log_mel_spectrogram_to_wav(spectrogram, name + '.wav')


def main():
    """ Main module if this file is invoked directly """
    # Load checkpoint
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--checkpoint", type=str, help="load a checkpoint")
    args = parser.parse_args()

    checkpoint_path = os.path.join(get_root_path(), args.checkpoint)
    checkpoint = load(os.path.join(get_root_path(), args.checkpoint))
    directory = os.path.dirname(checkpoint_path)
    logger.info('Loaded checkpoint: %s (%d Step)' % (args.checkpoint, checkpoint['step']))
    text_encoder = checkpoint['text_encoder']

    model = checkpoint['model']
    model.apply(lambda m: m.flatten_parameters() if hasattr(m, 'flatten_parameters') else None)
    torch.set_grad_enabled(False)
    model.train(mode=False)

    logger.info('Total Parameters: %d', get_total_parameters(model))
    logger.info('Model:\n%s' % model)

    while True:
        text = input("Text: ")
        text = text_encoder.encode(text).unsqueeze(0)
        frames, frames_with_residual, stop_tokens, alignments = model(text)
        sample_spectrogram(frames_with_residual, directory, 'generated_predicted_post_spectrogram')
        sample_spectrogram(frames, directory, 'generated_predicted_pre_spectrogram')


if __name__ == '__main__':
    main()

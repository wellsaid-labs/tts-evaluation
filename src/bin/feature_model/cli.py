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

from src.utils import get_total_parameters
from src.hparams import set_hparams
from src.bin.feature_model._utils import sample_attention
from src.bin.feature_model._utils import sample_spectrogram
from src.bin.feature_model._utils import load_checkpoint


def main():  # pragma: no cover
    """ Main module if this file is invoked directly """
    set_hparams()

    # Load checkpoint
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--checkpoint", type=str, help="load a checkpoint")
    args = parser.parse_args()

    checkpoint = load_checkpoint(args.checkpoint)
    directory = os.path.dirname(args.checkpoint)
    logger.info('Loaded checkpoint: %s (%d step)' % (args.checkpoint, checkpoint['step']))
    text_encoder = checkpoint['text_encoder']
    model = checkpoint['model']
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
        sample_spectrogram(frames_with_residual,
                           os.path.join(directory, 'generated_predicted_post_spectrogram'))
        sample_spectrogram(frames, os.path.join(directory, 'generated_predicted_pre_spectrogram'))
        sample_attention(alignments, os.path.join(directory, 'generated_attention'))
        print('–' * 100)


if __name__ == '__main__':  # pragma: no cover
    main()

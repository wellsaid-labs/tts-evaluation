import argparse
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import torch

from src.audio import griffin_lim
from src.bin.feature_model._utils import load_checkpoint
from src.bin.feature_model._utils import set_hparams as set_feature_model_auxiliary_hparams
from src.hparams import set_hparams
from src.utils import get_total_parameters
from src.utils import plot_attention
from src.utils import plot_spectrogram
from src.utils import plot_stop_token


def main(checkpoint):  # pragma: no cover
    """ Main module if this file is invoked directly """
    set_hparams()
    set_feature_model_auxiliary_hparams()

    checkpoint = load_checkpoint(checkpoint)
    directory = os.path.dirname(checkpoint)
    logger.info('Loaded checkpoint: %s (%d step)' % (checkpoint, checkpoint['step']))
    text_encoder = checkpoint['text_encoder']
    model = checkpoint['model']
    torch.set_grad_enabled(False)
    model.train(mode=False)

    logger.info('Total Parameters: %d', get_total_parameters(model))
    logger.info('Model:\n%s' % model)

    while True:
        text = input('Text: ').strip()
        logger.info('Got text: %s', text)

        text = text_encoder.encode(text.strip()).unsqueeze(0)

        logger.info('Predicting mel-spectrogram...')
        (predicted_pre_frames, predicted_post_frames, predicted_stop_tokens,
         predicted_alignments) = model(
             text, max_recursion=10000)

        logger.info('Plotting graphs...')
        plot_spectrogram(predicted_post_frames[:, 0]).savefig(
            os.path.join(directory, 'predicted_post_spectrogram.png'))
        plot_spectrogram(predicted_pre_frames[:, 0]).savefig(
            os.path.join(directory, 'predicted_pre_spectrogram.png'))
        plot_attention(predicted_alignments[:, 0]).savefig(
            os.path.join(directory, 'predicted_aligments.png'))
        plot_stop_token(predicted_stop_tokens[:, 0]).savefig(
            os.path.join(directory, 'predicted_stop_token.png'))

        logger.info('Running Griffin-lim...')
        griffin_lim(predicted_post_frames[:, 0], os.path.join(directory, 'predicted_audio.wav'))

        print('–' * 100)


if __name__ == '__main__':  # pragma: no cover
    # Load checkpoint
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--checkpoint", type=str, help="load a checkpoint", required=True)
    args = parser.parse_args()

    main(args.checkpoint)

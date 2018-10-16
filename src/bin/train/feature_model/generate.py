""" Generate relevant training files for the signal model.
"""
from pathlib import Path

import argparse
import logging
import sys

from tqdm import tqdm
from torch.nn import MSELoss
from torchnlp.utils import pad_batch

import torch
import numpy as np

from src.bin.train.feature_model._data_iterator import DataIterator
from src.bin.train.feature_model._utils import load_data
from src.bin.train.feature_model._utils import set_hparams
from src.utils import get_total_parameters
from src.utils import load_checkpoint
from src.utils import duplicate_stream
from src.utils.configurable import configurable
from src.utils.configurable import log_config

logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)


def _compute_loss_frame_loss(batch, predicted_post_frames, criterion_frames):  # pragma: no cover
    """ Compute the frame loss for Tacotron.

    Args:
        batch (dict): ``dict`` from ``src.bin.train.feature_model._utils.DataIterator``.
        predicted_post_frames (torch.FloatTensor [num_frames, batch_size, frame_channels]):
            Predicted frames with residual.
        criterion_frames (callable): Loss function used to score frame predictions.

    Returns:
        post_frames_loss (torch.Tensor [scalar])
        num_frame_predictions (int): Number of realized frame predictions taking masking into
            account.
    """
    mask = [torch.FloatTensor(length).fill_(1) for length in batch['frame_lengths']]
    mask = pad_batch(mask)[0].to(predicted_post_frames.device).transpose(0, 1)
    # [num_frames, batch_size] → [num_frames, batch_size, frame_channels]
    mask = mask.unsqueeze(2).expand_as(batch['frames'])

    num_predictions = torch.sum(mask)

    loss = criterion_frames(predicted_post_frames, batch['frames'])
    loss = torch.sum(loss * mask)

    return loss.item(), num_predictions.item()


@configurable
def main(checkpoint,
         destination='data/.signal_dataset/',
         destination_train='train',
         destination_dev='dev',
         destination_stdout='stdout.log',
         destination_stderr='stderr.log',
         max_batch_size=96,
         num_workers=1):  # pragma: no cover
    """ Main module used to generate dataset for training a signal model.

    Args:
        checkpoint (str): Checkpoint to load used to generate.
        destination (str, optional): Directory to save generated files to be used for
            training.
        destination_train (str, optional): Directory to save generated files to be used for
            training.
        destination_dev (str, optional): Directory to save generated files to be used for
            development.
        destination_stdout (str, optional): Filename to save stderr logs in.
        destination_stderr (str, optional): Filename to save stdout logs in.
        max_batch_size (int, optional): Maximum batch size predicted at a time.
        num_workers (int, optional): Number of workers for data loading.
    """
    destination = Path(destination)
    destination_train = destination / destination_train
    destination_train.mkdir(parents=True, exist_ok=True)

    destination_dev = destination / destination_dev
    destination_dev.mkdir(parents=True, exist_ok=True)

    duplicate_stream(sys.stdout, destination / destination_stdout)
    duplicate_stream(sys.stderr, destination / destination_stderr)

    log_config()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    checkpoint = load_checkpoint(checkpoint, device)
    text_encoder = checkpoint['text_encoder']
    model = checkpoint['model']
    train, dev, text_encoder = load_data(text_encoder=text_encoder, load_signal=True)

    # Check to ensure the the spectrogram loss is similar
    criterion_frames = MSELoss(reduction='none').to(device)

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
        logger.info('Generating for %s', destination)
        total_loss, total_predictions = 0.0, 0
        data_iterator = tqdm(
            DataIterator(
                device=device,
                dataset=dataset,
                batch_size=max_batch_size,
                load_signal=True,
                num_workers=num_workers))
        for i, batch in enumerate(data_iterator):

            _, batch_predicted_frames, _, _ = model(
                batch['text'], ground_truth_frames=batch['frames'])

            # NOTE: Compute loss to ensure consistency
            loss, num_predictions = _compute_loss_frame_loss(batch, batch_predicted_frames,
                                                             criterion_frames)
            total_predictions += num_predictions
            total_loss += loss

            # [num_frames, batch_size, frame_channels] → [batch_size, num_frames, frame_channels]
            batch_predicted_frames = batch_predicted_frames.transpose(0, 1).cpu().numpy().astype(
                np.float32)
            batch_size = batch_predicted_frames.shape[0]

            # Save predictions
            for j in range(batch_size):
                # [batch_size, num_frames, frame_channels] → [num_frames, frame_channels]
                predicted_frames = batch_predicted_frames[j][:batch['frame_lengths'][j]]
                # [batch_size, signal_length] → [signal_length]
                signal = batch['signal'][j].cpu().numpy().astype(np.float32)
                assert signal.shape[0] % predicted_frames.shape[0] == 0

                # NOTE: ``numpy .npy (no pickle)`` is about 15 times faster than numpy with pickle
                # ``torch.save`` does not include none-pickle options; therefore, we use numpy
                # Performance is important here because we are saving 10,000+ files.
                # REFERENCE: https://github.com/mverleg/array_storage_benchmark
                np.save(
                    str(destination / ('log_mel_spectrogram_%d_%d.npy' % (i, j))),
                    predicted_frames,
                    allow_pickle=False)
                np.save(
                    str(destination / ('signal_%d_%d.npy' % (i, j))), signal, allow_pickle=False)

        logger.info('Sanity check, post frame loss: %f [%f of %d]', total_loss / total_predictions,
                    total_loss, total_predictions)


if __name__ == '__main__':  # pragma: no cover
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c',
        '--checkpoint',
        type=str,
        default=None,
        help='Load a checkpoint from a path',
        required=True)
    args = parser.parse_args()

    set_hparams()

    main(checkpoint=args.checkpoint)

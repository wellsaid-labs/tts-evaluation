import random

from torch.utils.data import DataLoader
from torchnlp.utils import pad_batch


class DataIterator(object):
    """ Get a batch iterator over the ``dataset``.

    Args:
        device (torch.device, optional): Device onto which to load data.
        dataset (list): Dataset to iterate over.
        batch_size (int): Size of the batch for iteration.
        trial_run (bool): If ``True``, the data iterator runs only one batch.
        num_workers (int, optional): Number of workers for data loading.
        random (random.Random, optional): Random number generator to sample data.
    """

    def __init__(self, device, dataset, batch_size, trial_run=False, num_workers=0, random=random):
        super().__init__()
        self.device = device

        random.shuffle(dataset)

        # ``drop_last`` to ensure full utilization of mutliple GPUs
        self.iterator = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=self._collate_fn,
            pin_memory=True,
            num_workers=num_workers,
            drop_last=True)
        self.trial_run = trial_run

    def _maybe_cuda(self, tensor):
        is_cuda = self.device.type == 'cuda'
        return tensor.cuda(device=self.device, non_blocking=True) if is_cuda else tensor

    def _collate_fn(self, batch):
        """ Collage function to turn a list of tensors into one batch tensor.

        Returns: (dict) with:
            * input_signal (torch.FloatTensor [batch_size, signal_length])
            * target_signal_coarse (torch.FloatTensor [batch_size, signal_length])
            * target_signal_fine (torch.FloatTensor [batch_size, signal_length])
            * signal_lengths (list): List of lengths for each signal.
            * log_mel_spectrogram (torch.FloatTensor [batch_size, num_frames, frame_channels])
            * log_mel_spectrogram_lengths (list): List of lengths for each spectrogram.
        """
        input_signal, signal_lengths = pad_batch([r['slice']['input_signal'] for r in batch])
        target_signal_coarse, _ = pad_batch([r['slice']['target_signal_coarse'] for r in batch])
        target_signal_fine, _ = pad_batch([r['slice']['target_signal_fine'] for r in batch])
        log_mel_spectrogram, log_mel_spectrogram_lengths = pad_batch(
            [r['slice']['log_mel_spectrogram'] for r in batch])

        return {
            'slice': {
                'input_signal': input_signal,
                'target_signal_coarse': target_signal_coarse,
                'target_signal_fine': target_signal_fine,
                'log_mel_spectrogram': log_mel_spectrogram,
                'log_mel_spectrogram_lengths': log_mel_spectrogram_lengths,
                'signal_lengths': signal_lengths,
            },
            'log_mel_spectrogram': [r['log_mel_spectrogram'] for r in batch],
            'signal': [r['signal'] for r in batch]
        }

    def __len__(self):
        return 1 if self.trial_run else len(self.iterator)

    def __iter__(self):
        for batch in self.iterator:
            slice_ = batch['slice']
            batch['slice']['log_mel_spectrogram'] = self._maybe_cuda(slice_['log_mel_spectrogram'])
            batch['slice']['target_signal_coarse'] = self._maybe_cuda(
                slice_['target_signal_coarse'])
            batch['slice']['target_signal_fine'] = self._maybe_cuda(slice_['target_signal_fine'])
            batch['slice']['input_signal'] = self._maybe_cuda(slice_['input_signal'])
            batch['log_mel_spectrogram'] = [
                self._maybe_cuda(s) for s in batch['log_mel_spectrogram']
            ]

            yield batch

            if self.trial_run:
                break

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
    """

    def __init__(self, device, dataset, batch_size, trial_run=False, num_workers=0):
        # ``drop_last`` to ensure full utilization of mutliple GPUs
        self.device = device
        self.iterator = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=self._collate_fn,
            pin_memory=True,
            num_workers=num_workers,
            drop_last=True)
        self.trial_run = trial_run

    def _maybe_cuda(self, tensor, **kwargs):
        return tensor.cuda(device=self.device, **kwargs) if self.device.type == 'cuda' else tensor

    def _collate_fn(self, batch):
        """ Collage function to turn a list of tensors into one batch tensor.

        Returns: (dict) with:
            * source_signals (torch.FloatTensor [batch_size, signal_length])
            * target_coarse_signals (torch.FloatTensor [batch_size, signal_length])
            * target_fine_signals (torch.FloatTensor [batch_size, signal_length])
            * signal_lengths (list): List of lengths for each signal.
            * frames (torch.FloatTensor [batch_size, num_frames, frame_channels])
            * spectrograms (list): List of spectrograms to be used for sampling.
        """
        source_signals, source_signal_lengths = pad_batch([r['source_signal_slice'] for r in batch])
        target_coarse_signals, target_signal_lengths = pad_batch(
            [r['target_signal_coarse_slice'] for r in batch])
        target_fine_signals, _ = pad_batch([r['target_signal_fine_slice'] for r in batch])
        frames, frames_lengths = pad_batch([r['frames_slice'] for r in batch])
        spectrograms = [r['log_mel_spectrogram'] for r in batch]
        signals = [r['signal'] for r in batch]
        length_diff = [s - t for s, t in zip(source_signal_lengths, target_signal_lengths)]
        assert length_diff.count(length_diff[0]) == len(length_diff), (
            "Source must be a constant amount longer than target; "
            "otherwise, they wont be aligned after padding.")
        return {
            'source_signals': source_signals,
            'target_coarse_signals': target_coarse_signals,
            'target_fine_signals': target_fine_signals,
            'target_signal_lengths': target_signal_lengths,
            'frames': frames,
            'spectrograms': spectrograms,
            'frames_lengths': frames_lengths,
            'signals': signals,
        }

    def __len__(self):
        return 1 if self.trial_run else len(self.iterator)

    def __iter__(self):
        for batch in self.iterator:
            batch['source_signals'] = self._maybe_cuda(batch['source_signals'], non_blocking=True)
            batch['target_coarse_signals'] = self._maybe_cuda(
                batch['target_coarse_signals'], non_blocking=True)
            batch['target_fine_signals'] = self._maybe_cuda(
                batch['target_fine_signals'], non_blocking=True)
            batch['frames'] = self._maybe_cuda(batch['frames'], non_blocking=True)
            batch['spectrograms'] = [
                self._maybe_cuda(s, non_blocking=True) for s in batch['spectrograms']
            ]

            yield batch

            if self.trial_run:
                break

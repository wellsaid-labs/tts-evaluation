from torch.utils.data import DataLoader
from torchnlp.samplers import BucketBatchSampler
from torchnlp.utils import pad_batch


class DataIterator(object):
    """ Get a batch iterator over the ``dataset``.

    Args:
        device (torch.device, optional): Device onto which to load data.
        dataset (list): Dataset to iterate over.
        batch_size (int): Iteration batch size.
        sort_key (callable): Sort key used to group similar length data in order to minimize
            padding; therefore, using less memory and speeding up training.
        trial_run (bool or int): If ``True``, iterates over one batch.
        load_signal (bool, optional): If ``True``, return signal during iteration.
        num_workers (int, optional): Number of workers for data loading.

    Returns:
        (torch.utils.data.DataLoader) Single-process or multi-process iterators over the dataset.
        Per iteration the batch returned includes:
            text (torch.LongTensor [batch_size, num_tokens])
            text_lengths (list): List of lengths for each sentence.
            frames (torch.FloatTensor [num_frames, batch_size, frame_channels])
            frame_lengths (list): List of lengths for each spectrogram.
            stop_token (torch.FloatTensor [num_frames, batch_size])
            signal (list): List of signals.
    """

    def __init__(self,
                 device,
                 dataset,
                 batch_size,
                 sort_key=lambda r: r['log_mel_spectrogram'].shape[0],
                 trial_run=False,
                 load_signal=False,
                 num_workers=0):
        batch_sampler = BucketBatchSampler(dataset, batch_size, False, sort_key=sort_key)
        self.device = device
        self.iterator = DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            collate_fn=self._collate_fn,
            pin_memory=True,
            num_workers=num_workers)
        self.trial_run = trial_run
        self.load_signal = load_signal

    def _maybe_cuda(self, tensor, **kwargs):
        return tensor.cuda(device=self.device, **kwargs) if self.device.type == 'cuda' else tensor

    def _collate_fn(self, batch):
        """ List of tensors to a batch variable """
        text_batch, text_length_batch = pad_batch([row['text'] for row in batch])
        frame_batch, frame_length_batch = pad_batch([row['log_mel_spectrogram'] for row in batch])
        stop_token_batch, _ = pad_batch([row['stop_token'] for row in batch])
        transpose = lambda b: b.transpose_(0, 1).contiguous()
        return {
            'text': text_batch,
            'text_lengths': text_length_batch,
            'frames': transpose(frame_batch),
            'frame_lengths': frame_length_batch,
            'stop_token': transpose(stop_token_batch),
            'signal': [row['signal'] for row in batch] if self.load_signal else None
        }

    def __len__(self):
        return 1 if self.trial_run else len(self.iterator)

    def __iter__(self):
        for batch in self.iterator:
            batch['text'] = self._maybe_cuda(batch['text'], non_blocking=True)
            batch['frames'] = self._maybe_cuda(batch['frames'], non_blocking=True)
            batch['stop_token'] = self._maybe_cuda(batch['stop_token'], non_blocking=True)
            if batch['signal'] is not None:
                batch['signal'] = [self._maybe_cuda(s, non_blocking=True) for s in batch['signal']]
            yield batch

            if self.trial_run:
                break

from concurrent.futures import ThreadPoolExecutor

import logging

from third_party.data_loader import DataLoader as DataBatchLoader
from torch.multiprocessing import cpu_count
from torch.utils import data
from torchnlp.samplers import BucketBatchSampler
from torchnlp.utils import pad_batch

import numpy as np
import torch
import tqdm

logger = logging.getLogger(__name__)


def get_spectrogram_length(filename):
    """ Get length of spectrogram (shape [num_frames, num_channels]) from a ``.npy`` numpy file

    Args:
        filename (str): Numpy file

    Returns:
        (int) Length of spectrogram
    """
    return np.load(str(filename)).shape[0]


class DataLoader(data.Dataset):
    """ DataLoader loads and preprocesses a single example.

    Args:
        data (iteratable of dict)
        text_encoder (torchnlp.TextEncoder): Text encoder used to encode and decode the text.
        text_key (str, optional): For each example in the data, this is the same of the text key.
        spectrogram_path_key (str, optional)
        other_keys (str, optional)
    """

    def __init__(self,
                 data,
                 text_encoder,
                 text_key='text',
                 spectrogram_path_key='spectrogram_path',
                 other_keys=['speaker']):
        self.data = data
        self.text_encoder = text_encoder
        self._spectrogram_lengths = None  # Cache for self.spectrogram_lengths
        self.text_key = text_key
        self.spectrogram_path_key = spectrogram_path_key
        self.other_keys = other_keys

    @property
    def spectrogram_lengths(self):
        if self._spectrogram_lengths is None:
            # NOTE: Spectrogram lengths for sorting
            logger.info('Computing spectrogram lengths...')
            with ThreadPoolExecutor() as pool:
                filenames = [row[self.spectrogram_path_key] for row in self.data]
                self._spectrogram_lengths = list(
                    tqdm.tqdm(pool.map(get_spectrogram_length, filenames), total=len(filenames)))

        return self._spectrogram_lengths

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        example = self.data[index]
        spectrogram = np.load(str(example[self.spectrogram_path_key]))
        spectrogram = torch.from_numpy(spectrogram)

        text = example[self.text_key].strip()
        text = self.text_encoder.encode(text)

        stop_token = spectrogram.new_zeros((spectrogram.shape[0],))
        stop_token[-1] = 1

        return_value = {
            'spectrogram': spectrogram,
            'stop_token': stop_token,
            'text': text,
        }
        for key in self.other_keys:
            return_value[key] = example[key]

        return return_value


class DataBatchIterator(object):
    """ Get a batch iterator over the ``data``.

    Args:
        data (iterable): Data to iterate over.
        text_encoder (torchnlp.TextEncoder): Text encoder used to encode and decode the text.
        batch_size (int): Iteration batch size.
        device (torch.device, optional): Device onto which to load data.
        trial_run (bool or int): If ``True``, iterates over one batch.
        num_workers (int, optional): Number of workers for data loading.

    Returns:
        (torch.utils.data.DataLoader) Single-process or multi-process iterators over the dataset.
        Per iteration the batch returned includes:
            text (torch.LongTensor [num_tokens, batch_size])
            text_lengths (list): List of lengths for each sentence.
            spectrogram (torch.FloatTensor [num_frames, batch_size, frame_channels])
            spectrogram_lengths (list): List of lengths for each spectrogram.
            spectrogram_expanded_mask (torch.FloatTensor [num_frames, batch_size, frame_channels])
            spectrogram_mask (torch.FloatTensor [num_frames, batch_size])
            stop_token (torch.FloatTensor [num_frames, batch_size])
    """

    def __init__(self,
                 data,
                 text_encoder,
                 batch_size,
                 device,
                 trial_run=False,
                 num_workers=cpu_count()):
        logger.info('Launching with %d workers', num_workers)
        data = DataLoader(data, text_encoder=text_encoder)
        batch_sampler = BucketBatchSampler(
            data.spectrogram_lengths,
            batch_size,
            drop_last=False,
            sort_key=lambda x: x,
            biggest_batches_first=lambda x: x)
        self.device = device
        self.iterator = DataBatchLoader(
            data,
            batch_sampler=batch_sampler,
            collate_fn=self._collate_fn,
            pin_memory=True,
            num_workers=num_workers)
        self.trial_run = trial_run

    def _maybe_cuda(self, tensor, **kwargs):
        return tensor.cuda(device=self.device, **kwargs) if self.device.type == 'cuda' else tensor

    def _collate_fn(self, batch):
        """ List of tensors to a batch variable """
        text, text_lengths = pad_batch([row['text'] for row in batch])
        spectrogram, spectrogram_lengths = pad_batch([row['spectrogram'] for row in batch])
        stop_token, _ = pad_batch([row['stop_token'] for row in batch])

        spectrogram_mask = [torch.FloatTensor(length).fill_(1) for length in spectrogram_lengths]
        spectrogram_mask, _ = pad_batch(spectrogram_mask)  # [batch_size, num_frames]

        text = text.transpose(0, 1).contiguous()
        spectrogram = spectrogram.transpose(0, 1).contiguous()
        stop_token = stop_token.transpose(0, 1).contiguous()
        spectrogram_mask = spectrogram_mask.transpose(0, 1).contiguous()  # [num_frames, batch_size]

        # [num_frames, batch_size] → [num_frames, batch_size, frame_channels]
        spectrogram_expanded_mask = spectrogram_mask.unsqueeze(2).expand_as(
            spectrogram).contiguous()

        return {
            'text': text,
            'text_lengths': text_lengths,
            'spectrogram': spectrogram,
            'spectrogram_lengths': spectrogram_lengths,
            'spectrogram_expanded_mask': spectrogram_expanded_mask,
            'spectrogram_mask': spectrogram_mask,
            'stop_token': stop_token,
        }

    def __len__(self):
        return 1 if self.trial_run else len(self.iterator)

    def __iter__(self):
        for batch in self.iterator:
            maybe_cuda_keys = [
                'text', 'spectrogram', 'spectrogram_expanded_mask', 'spectrogram_mask', 'stop_token'
            ]
            for key in maybe_cuda_keys:
                batch[key] = self._maybe_cuda(batch[key], non_blocking=True)

            yield batch

            if self.trial_run:
                break

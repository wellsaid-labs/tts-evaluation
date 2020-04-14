from collections import defaultdict
from functools import partial

import logging
import math

from torch.nn.functional import mse_loss
from torchnlp.encoders.text import stack_and_pad_tensors
from torchnlp.utils import collate_tensors
from torchnlp.utils import lengths_to_mask
from torchnlp.utils import tensors_to

import torch
import torch.utils.data

from src.utils.averaged_metric import AveragedMetric
from src.utils.data_loader import DataLoader
from src.utils.on_disk_tensor import cache_on_disk_tensor_shapes
from src.utils.on_disk_tensor import maybe_load_tensor
from src.utils.on_disk_tensor import OnDiskTensor
from src.utils.utils import evaluate
from src.utils.utils import get_average_norm
from src.utils.utils import get_weighted_stdev
from src.utils.utils import sort_together

logger = logging.getLogger(__name__)

# LEARN MORE: https://github.com/pytorch/pytorch/issues/973
torch.multiprocessing.set_sharing_strategy('file_system')


def _batch_predict_spectrogram_load_fn(row, input_encoder, load_spectrogram=False):
    """ Load function for loading a single row.

    Args:
        row (TextSpeechRow)
        input_encoder (src.spectrogram_model.InputEncoder): Spectrogram model input encoder.
        load_spectrogram (bool, optional)

    Returns:
        (TextSpeechRow)
    """
    encoded_text, encoded_speaker = input_encoder.encode((row.text, row.speaker))
    row = row._replace(text=encoded_text, speaker=encoded_speaker)
    if load_spectrogram:
        row = row._replace(spectrogram=maybe_load_tensor(row.spectrogram))
    return row


def batch_predict_spectrograms(data,
                               input_encoder,
                               model,
                               device,
                               batch_size,
                               filenames=None,
                               aligned=True,
                               use_tqdm=True):
    """ Batch predict spectrograms.

    TODO: Following running this function, for some reason, the `signal_model.trainer` runs out
    of memory.

    Args:
        data (iterable of TextSpeechRow)
        input_encoder (src.spectrogram_model.InputEncoder): Spectrogram model input encoder.
        model (torch.nn.Module): Model used to compute spectrograms.
        batch_size (int)
        device (torch.device): Device to run model on.
        filenames (list, optional): If provided, this saves predictions to these paths.
        aligned (bool, optional): If ``True``, predict a ground truth aligned spectrogram.
        use_tqdm (bool, optional): Write a progress bar to standard streams.

    Returns:
        (iterable of torch.Tensor or OnDiskTensor)
    """
    data = [d._replace(metadata={'index': i}) for i, d in enumerate(data)]
    if filenames is not None:
        assert len(filenames) == len(data)
        iterator = zip(data, filenames)
        data = [d._replace(metadata=dict({'filename': f}, **d.metadata)) for d, f in iterator]
    return_ = [None] * len(data)

    # NOTE: Sort by sequence length to reduce padding in batches.
    # NOTE: Sort by longest to shortest allowing us to trigger any OOM errors earlier.
    if all([r.spectrogram is not None for r in data]):
        cache_on_disk_tensor_shapes(
            [r.spectrogram for r in data if isinstance(r.spectrogram, OnDiskTensor)])
        data = sort_together(data, [-r.spectrogram.shape[0] for r in data])
    else:
        data = sorted(data, key=lambda r: len(r.text), reverse=True)

    load_fn_partial = partial(
        _batch_predict_spectrogram_load_fn, input_encoder=input_encoder, load_spectrogram=aligned)
    loader = DataLoader(
        data,
        batch_size=batch_size,
        load_fn=load_fn_partial,
        post_processing_fn=partial(tensors_to, device=device, non_blocking=True),
        collate_fn=partial(collate_tensors, stack_tensors=partial(stack_and_pad_tensors, dim=1)),
        pin_memory=True,
        use_tqdm=use_tqdm)
    with evaluate(model, device=device):
        metrics = defaultdict(AveragedMetric)
        for batch in loader:
            # Predict spectrogram
            text, text_lengths = batch.text
            speaker = batch.speaker[0]
            if aligned:
                spectrogram, spectrogram_lengths = batch.spectrogram
                _, predictions, _, alignments = model(text, speaker, text_lengths, spectrogram,
                                                      spectrogram_lengths)
            else:
                _, predictions, _, alignments, spectrogram_lengths, _ = model(
                    text, speaker, text_lengths)

            # Compute metrics for logging
            mask = lengths_to_mask(spectrogram_lengths, device=predictions.device).transpose(0, 1)
            metrics['attention_norm'].update(
                get_average_norm(alignments, norm=math.inf, dim=2, mask=mask), mask.sum())
            metrics['attention_std'].update(
                get_weighted_stdev(alignments, dim=2, mask=mask), mask.sum())
            if aligned:
                mask = mask.unsqueeze(2).expand_as(predictions)
                loss = mse_loss(predictions, spectrogram, reduction='none')
                metrics['loss'].update(loss.masked_select(mask).mean(), mask.sum())

            # Split batch and store in-memory or on-disk
            spectrogram_lengths = spectrogram_lengths.squeeze(0).tolist()
            predictions = predictions.split(1, dim=1)
            predictions = [p[:l, 0] for p, l in zip(predictions, spectrogram_lengths)]
            for i, prediction in enumerate(predictions):
                return_index = batch.metadata['index'][i]
                if filenames is None:
                    return_[return_index] = prediction
                else:
                    filename = batch.metadata['filename'][i]
                    return_[return_index] = OnDiskTensor.from_tensor(filename, prediction)

        for name, metric in metrics.items():
            logger.info('Prediction metric (%s): %s', name, metric.reset())

    return return_

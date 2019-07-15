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

from src.utils.accumulated_metrics import AccumulatedMetrics
from src.utils.data_loader import DataLoader
from src.utils.on_disk_tensor import OnDiskTensor
from src.utils.utils import evaluate
from src.utils.utils import get_average_norm
from src.utils.utils import get_tensors_dim_length
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
    if load_spectrogram and isinstance(row.spectrogram, OnDiskTensor):
        row = row._replace(spectrogram=row.spectrogram.to_tensor())
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

    # Sort by sequence length to reduce padding in batches.
    if all([r.spectrogram is not None for r in data]):
        spectrogram_lengths = get_tensors_dim_length([r.spectrogram for r in data])
        data = sort_together(data, spectrogram_lengths)
    else:
        data = sorted(data, key=lambda r: len(r.text))

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
        metrics = AccumulatedMetrics()
        for batch in loader:
            # Predict spectrogram
            text, text_lengths = batch.text
            speaker = batch.speaker[0]
            if aligned:
                spectrogram, spectrogram_lengths = batch.spectrogram
                _, predictions, _, alignments = model(text, speaker, text_lengths, spectrogram,
                                                      spectrogram_lengths)
            else:
                _, predictions, _, alignments, spectrogram_lengths = model(text, speaker)

            # Compute metrics for logging
            mask = lengths_to_mask(spectrogram_lengths, device=predictions.device).transpose(0, 1)
            metrics.add_metrics({
                'attention_norm': get_average_norm(alignments, norm=math.inf, dim=2, mask=mask),
                'attention_std': get_weighted_stdev(alignments, dim=2, mask=mask),
            }, mask.sum())
            if aligned:
                mask = mask.unsqueeze(2).expand_as(predictions)
                loss = mse_loss(predictions, spectrogram, reduction='none')
                metrics.add_metric('loss', loss.masked_select(mask).mean(), mask.sum())

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

        metrics.log_epoch_end(lambda k, v: logger.info('Prediction metric (%s): %s', k, v))

    return return_

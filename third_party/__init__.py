import math

import torch


def pin_memory_batch(batch):
    # Taken from: https://github.com/pytorch/pytorch/blob/v1.0.1/torch/utils/data/dataloader.py#L237
    if isinstance(batch, torch.Tensor):
        return batch.pin_memory()
    elif isinstance(batch, torch._six.string_classes):
        return batch
    # TODO: Send a PR to PyTorch GitHub concerning this.
    # CHANGED: This branch was added because ``container_abcs.Sequence`` was changing a
    # ``namedtuple`` to a ``list``.
    # Inspired by:
    # https://stackoverflow.com/questions/2166818/how-to-check-if-an-object-is-an-instance-of-a-namedtuple
    elif hasattr(batch, '_asdict') and isinstance(batch, tuple):  # Handle ``namedtuple``
        return batch.__class__(**pin_memory_batch(batch._asdict()))
    elif isinstance(batch, torch._six.container_abcs.Mapping):
        return {k: pin_memory_batch(sample) for k, sample in batch.items()}
    elif isinstance(batch, torch._six.container_abcs.Sequence):
        return [pin_memory_batch(sample) for sample in batch]
    else:
        return batch


def get_parameter_norm(parameters, norm_type=2):
    """ Compute the total norm of the parameters.

    The norm is computed over all gradients together, as if they were
    concatenated into a single vector.

    Taken from:
    https://pytorch.org/docs/stable/_modules/torch/nn/utils/clip_grad.html#clip_grad_norm_

    Args:
        parameters (Iterable[Tensor]): An iterable of Tensors.
        norm_type (float or int, optional): Type of the used p-norm. Can be ``'inf'`` for infinity
            norm.

    Return:
        Total norm of the parameters (viewed as a single vector).
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    if norm_type == math.inf:
        total_norm = max(p.grad.data.abs().max() for p in parameters)
    else:
        total_norm = 0
        for p in parameters:
            param_norm = p.grad.data.norm(norm_type)
            total_norm += param_norm.item()**norm_type
        total_norm = total_norm**(1. / norm_type)
    return total_norm

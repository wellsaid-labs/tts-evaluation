import math

import torch


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

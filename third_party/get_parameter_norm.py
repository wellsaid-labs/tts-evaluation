# fmt: off
# type: ignore
import math

import torch


def get_parameter_norm(parameters, norm_type: float = 2) -> torch.Tensor:
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
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.0)
    device = parameters[0].grad.device
    if norm_type == math.inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(
            torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]),
            norm_type,
        )
    return total_norm

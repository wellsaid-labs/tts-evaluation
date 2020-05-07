from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import logging

from tqdm import tqdm

import numpy as np
import torch
import torch.utils.data

from src.utils.disk_cache_ import disk_cache
from src.utils.disk_cache_ import make_arg_key
from src.utils.utils import log_runtime

logger = logging.getLogger(__name__)


@disk_cache
def _get_on_disk_tensor_shape(path):
    if not path.is_file():
        raise RuntimeError('Tensor not found on disk.')

    with open(str(path), 'rb') as file_:
        try:
            version = np.lib.format.read_magic(file_)
        except ValueError as error:
            logger.error('Failed to read shape of %s' % file_)
            raise error
        shape, _, _ = np.lib.format._read_array_header(file_, version)

    return shape


@log_runtime
def cache_on_disk_tensor_shapes(tensors):
    tensors = [
        t for t in tensors if make_arg_key(_get_on_disk_tensor_shape.__wrapped__, t.path) not in
        _get_on_disk_tensor_shape.disk_cache
    ]
    if len(tensors) == 0:
        return

    logger.info('Caching `OnDiskTensor` shape metadata for %d tensors.', len(tensors))
    with ThreadPoolExecutor() as pool:
        iterator = pool.map(_get_on_disk_tensor_shape, [t.path for t in tensors])
        iterator = tqdm(iterator, total=len(tensors))
        list(iterator)

    _get_on_disk_tensor_shape.disk_cache.save()


def maybe_load_tensor(tensor):
    """ Load `tensor` into memory if it's not in memory already.

    Args:
        tensor (torch.Tensor or OnDiskTensor)

    Returns:
        (torch.Tensor)
    """
    return tensor.to_tensor() if isinstance(tensor, OnDiskTensor) else tensor


class OnDiskTensor():
    """ Tensor that resides on disk.

    TODO: Consider implementing an `onDiskTensor` by overriding `torch.Tensor` via:
    https://github.com/pytorch/pytorch/issues/22402

    Args:
        path (str or Path): Path to a tensor saved on disk as an ``.npy`` file.
        allow_pickle (bool, optional): Allow saving object arrays using Python pickles. This
          is not recommended for performance reasons.
    """

    def __init__(self, path, allow_pickle=False):
        assert '.npy' in str(path), 'Path must include ``.npy`` extension.'

        self.path = Path(path)
        self.allow_pickle = allow_pickle

    def __hash__(self):
        return hash(self.path)

    def __eq__(self, other):
        if isinstance(other, OnDiskTensor):
            return self.path == other.path

        # Learn more:
        # https://stackoverflow.com/questions/878943/why-return-notimplemented-instead-of-raising-notimplementederror
        return NotImplemented

    @property
    def shape(self):
        return _get_on_disk_tensor_shape(self.path)

    def to_tensor(self):
        """ Convert to a in-memory ``torch.tensor``. """
        if not self.exists():
            raise RuntimeError('Tensor not found on disk.')

        loaded = np.load(str(self.path), allow_pickle=self.allow_pickle)
        return torch.from_numpy(loaded).contiguous()

    def exists(self):
        """ If ``True``, the tensor exists on disk. """
        return self.path.is_file()

    def unlink(self):
        """ Delete the ``OnDiskTensor`` from disk.

        Returns:
            (Path): The path the ``OnDiskTensor`` used to reside in.
        """
        self.path.unlink()
        return self

    @classmethod
    def from_tensor(class_, path, tensor, allow_pickle=False):
        """ Make a ``OnDiskTensor`` from a tensor.

        Args:
            path (str or Path): Path to a tensor saved on disk as an ``.npy`` file.
            tensor (np.array or torch.tensor)
            allow_pickle (bool, optional): Allow saving object arrays using Python pickles. This
              is not recommended for performance reasons.
        """
        if torch.is_tensor(tensor):
            tensor = tensor.cpu().numpy()

        # This storage was picked using this benchmark:
        # https://github.com/mverleg/array_storage_benchmark
        np.save(str(path), tensor, allow_pickle=allow_pickle)
        return class_(path, allow_pickle)

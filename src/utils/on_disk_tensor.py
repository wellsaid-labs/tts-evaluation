from pathlib import Path

import logging

import numpy as np
import torch
import torch.utils.data

logger = logging.getLogger(__name__)


class OnDiskTensor():
    """ Tensor that resides on disk.

    Args:
        path (str or Path): Path to a tensor saved on disk as an ``.npy`` file.
        allow_pickle (bool, optional): Allow saving object arrays using Python pickles. This
          is not recommended for performance reasons.
    """

    def __init__(self, path, allow_pickle=False):
        assert '.npy' in str(path), 'Path must include ``.npy`` extension.'

        self.path = Path(path)
        self.allow_pickle = allow_pickle
        self._shape = None
        self._shape_path = None

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
        if not self.exists():
            raise RuntimeError('Tensor not found on disk.')

        if self.path == self._shape_path:
            return self._shape

        with open(str(self.path), 'rb') as file_:
            try:
                version = np.lib.format.read_magic(file_)
            except ValueError as error:
                logger.error('Failed to read shape of %s' % file_)
                raise error
            shape, _, _ = np.lib.format._read_array_header(file_, version)

        self._shape = shape
        self._shape_path = self.path

        return shape

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

""" Borrowed from https://github.com/pytorch/pytorch with a fix for
https://github.com/pytorch/pytorch/pull/9804
"""
# flake8: noqa
from copy import deepcopy
from functools import wraps
from itertools import product
from numbers import Number
from torch import multiprocessing as mp
from torch._six import string_classes, inf
from torch._utils_internal import get_writable_path
from src.utils import DataLoader
from torch.utils.data import Dataset, TensorDataset, ConcatDataset
from torch.utils.data.dataloader import default_collate, ExceptionWrapper
from torch.utils.data.dataloader import MANAGER_STATUS_CHECK_INTERVAL
from torch.utils.data.dataset import random_split
import __main__
import argparse
import contextlib
import ctypes
import errno
import gc
import inspect
import math
import os
import platform
import random
import re
import signal
import subprocess
import sys
import time
import torch
import torch.backends.cudnn
import torch.backends.mkl
import torch.cuda
import types
import unittest
import warnings

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--accept', action='store_true')
args, remaining = parser.parse_known_args()
SEED = args.seed
ACCEPT = args.accept
UNITTEST_ARGS = [sys.argv[0]] + remaining
torch.manual_seed(SEED)


def run_tests(argv=UNITTEST_ARGS):
    unittest.main(argv=argv)


PY3 = sys.version_info > (3, 0)
PY34 = sys.version_info >= (3, 4)

IS_WINDOWS = sys.platform == "win32"
IS_PPC = platform.machine() == "ppc64le"


def _check_module_exists(name):
    r"""Returns if a top-level module with :attr:`name` exists *without**
    importing it. This is generally safer than try-catch block around a
    `import X`. It avoids third party libraries breaking assumptions of some of
    our tests, e.g., setting multiprocessing start method when imported
    (see librosa/#747, torchvision/#544).
    """
    if not PY3:  # Python 2
        import imp
        try:
            imp.find_module(name)
            return True
        except ImportError:
            return False
    elif PY34:  # Python [3, 3.4)
        import importlib
        loader = importlib.find_loader(name)
        return loader is not None
    else:  # Python >= 3.4
        import importlib
        spec = importlib.util.find_spec(name)
        return spec is not None


TEST_NUMPY = _check_module_exists('numpy')
TEST_SCIPY = _check_module_exists('scipy')
TEST_MKL = torch.backends.mkl.is_available()

# On Py2, importing librosa 0.6.1 triggers a TypeError (if using newest joblib)
# see librosa/librosa#729.
# TODO: allow Py2 when librosa 0.6.2 releases
TEST_LIBROSA = _check_module_exists('librosa') and PY3

NO_MULTIPROCESSING_SPAWN = os.environ.get('NO_MULTIPROCESSING_SPAWN', '0') == '1'
TEST_WITH_ASAN = os.getenv('PYTORCH_TEST_WITH_ASAN', '0') == '1'
TEST_WITH_UBSAN = os.getenv('PYTORCH_TEST_WITH_UBSAN', '0') == '1'
TEST_WITH_ROCM = os.getenv('PYTORCH_TEST_WITH_ROCM', '0') == '1'

if TEST_NUMPY:
    import numpy


def skipIfRocm(fn):

    @wraps(fn)
    def wrapper(*args, **kwargs):
        if TEST_WITH_ROCM:
            raise unittest.SkipTest("test doesn't currently work on the ROCm stack")
        else:
            fn(*args, **kwargs)

    return wrapper


def skipIfNoLapack(fn):

    @wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            fn(*args, **kwargs)
        except Exception as e:
            if 'Lapack library not found' in repr(e):
                raise unittest.SkipTest('Compiled without Lapack')
            raise

    return wrapper


def skipCUDAMemoryLeakCheckIf(condition):

    def dec(fn):
        if getattr(fn, '_do_cuda_memory_leak_check', True):  # if current True
            fn._do_cuda_memory_leak_check = not condition
        return fn

    return dec


def get_cuda_memory_usage():
    # we don't need CUDA synchronize because the statistics are not tracked at
    # actual freeing, but at when marking the block as free.
    num_devices = torch.cuda.device_count()
    gc.collect()
    return tuple(torch.cuda.memory_allocated(i) for i in range(num_devices))


def suppress_warnings(fn):

    @wraps(fn)
    def wrapper(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fn(*args, **kwargs)

    return wrapper


def get_cpu_type(type_name):
    module, name = type_name.rsplit('.', 1)
    assert module == 'torch.cuda'
    return getattr(torch, name)


def get_gpu_type(type_name):
    if isinstance(type_name, type):
        type_name = '{}.{}'.format(type_name.__module__, type_name.__name__)
    module, name = type_name.rsplit('.', 1)
    assert module == 'torch'
    return getattr(torch.cuda, name)


def to_gpu(obj, type_map={}):
    if isinstance(obj, torch.Tensor):
        assert obj.is_leaf
        t = type_map.get(obj.type(), get_gpu_type(obj.type()))
        with torch.no_grad():
            res = obj.clone().type(t)
            res.requires_grad = obj.requires_grad
        return res
    elif torch.is_storage(obj):
        return obj.new().resize_(obj.size()).copy_(obj)
    elif isinstance(obj, list):
        return [to_gpu(o, type_map) for o in obj]
    elif isinstance(obj, tuple):
        return tuple(to_gpu(o, type_map) for o in obj)
    else:
        return deepcopy(obj)


def get_function_arglist(func):
    return inspect.getargspec(func).args


def set_rng_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    if TEST_NUMPY:
        numpy.random.seed(seed)


@contextlib.contextmanager
def freeze_rng_state():
    rng_state = torch.get_rng_state()
    if torch.cuda.is_available():
        cuda_rng_state = torch.cuda.get_rng_state()
    yield
    if torch.cuda.is_available():
        torch.cuda.set_rng_state(cuda_rng_state)
    torch.set_rng_state(rng_state)


def iter_indices(tensor):
    if tensor.dim() == 0:
        return range(0)
    if tensor.dim() == 1:
        return range(tensor.size(0))
    return product(*(range(s) for s in tensor.size()))


def is_iterable(obj):
    try:
        iter(obj)
        return True
    except TypeError:
        return False


class TestCase(unittest.TestCase):
    precision = 1e-5
    maxDiff = None
    _do_cuda_memory_leak_check = False

    def __init__(self, method_name='runTest'):
        super(TestCase, self).__init__(method_name)
        # Wraps the tested method if we should do CUDA memory check.
        test_method = getattr(self, method_name)
        self._do_cuda_memory_leak_check &= getattr(test_method, '_do_cuda_memory_leak_check', True)
        # FIXME: figure out the flaky -1024 anti-leaks on windows. See #8044
        if self._do_cuda_memory_leak_check and not IS_WINDOWS:
            # the import below may initialize CUDA context, so we do it only if
            # self._do_cuda_memory_leak_check is True.
            from common_cuda import TEST_CUDA
            fullname = self.id().lower()  # class_name.method_name
            if TEST_CUDA and ('gpu' in fullname or 'cuda' in fullname):
                # initialize context & RNG to prevent false positive detections
                # when the test is the first to initialize those
                from common_cuda import initialize_cuda_context_rng
                initialize_cuda_context_rng()
                setattr(self, method_name, self.wrap_with_cuda_memory_check(test_method))

    def wrap_with_cuda_memory_check(self, method):
        # Assumes that `method` is the tested function in `self`.
        # NOTE: Python Exceptions (e.g., unittest.Skip) keeps objects in scope
        #       alive, so this cannot be done in setUp and tearDown because
        #       tearDown is run unconditionally no matter whether the test
        #       passes or not. For the same reason, we can't wrap the `method`
        #       call in try-finally and always do the check.
        @wraps(method)
        def wrapper(self, *args, **kwargs):
            befores = get_cuda_memory_usage()
            method(*args, **kwargs)
            afters = get_cuda_memory_usage()
            for i, (before, after) in enumerate(zip(befores, afters)):
                self.assertEqual(
                    before, after, '{} leaked {} bytes CUDA memory on device {}'.format(
                        self.id(), after - before, i))

        return types.MethodType(wrapper, self)

    def setUp(self):
        set_rng_seed(SEED)

    def assertTensorsSlowEqual(self, x, y, prec=None, message=''):
        max_err = 0
        self.assertEqual(x.size(), y.size())
        for index in iter_indices(x):
            max_err = max(max_err, abs(x[index] - y[index]))
        self.assertLessEqual(max_err, prec, message)

    def safeToDense(self, t):
        r = self.safeCoalesce(t)
        return r.to_dense()

    def safeCoalesce(self, t):
        tc = t.coalesce()
        self.assertEqual(tc.to_dense(), t.to_dense())
        self.assertTrue(tc.is_coalesced())

        # Our code below doesn't work when nnz is 0, because
        # then it's a 0D tensor, not a 2D tensor.
        if t._nnz() == 0:
            self.assertEqual(t._indices(), tc._indices())
            self.assertEqual(t._values(), tc._values())
            return tc

        value_map = {}
        for idx, val in zip(t._indices().t(), t._values()):
            idx_tup = tuple(idx.tolist())
            if idx_tup in value_map:
                value_map[idx_tup] += val
            else:
                value_map[idx_tup] = val.clone() if isinstance(val, torch.Tensor) else val

        new_indices = sorted(list(value_map.keys()))
        new_values = [value_map[idx] for idx in new_indices]
        if t._values().ndimension() < 2:
            new_values = t._values().new(new_values)
        else:
            new_values = torch.stack(new_values)

        new_indices = t._indices().new(new_indices).t()
        tg = t.new(new_indices, new_values, t.size())

        self.assertEqual(tc._indices(), tg._indices())
        self.assertEqual(tc._values(), tg._values())

        if t.is_coalesced():
            self.assertEqual(tc._indices(), t._indices())
            self.assertEqual(tc._values(), t._values())

        return tg

    def assertEqual(self, x, y, prec=None, message='', allow_inf=False):
        if isinstance(prec, str) and message == '':
            message = prec
            prec = None
        if prec is None:
            prec = self.precision

        if isinstance(x, torch.Tensor) and isinstance(y, Number):
            self.assertEqual(x.item(), y, prec, message, allow_inf)
        elif isinstance(y, torch.Tensor) and isinstance(x, Number):
            self.assertEqual(x, y.item(), prec, message, allow_inf)
        elif isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor):

            def assertTensorsEqual(a, b):
                super(TestCase, self).assertEqual(a.size(), b.size(), message)
                if a.numel() > 0:
                    b = b.type_as(a)
                    b = b.cuda(device=a.get_device()) if a.is_cuda else b.cpu()
                    # check that NaNs are in the same locations
                    nan_mask = a != a
                    self.assertTrue(torch.equal(nan_mask, b != b), message)
                    diff = a - b
                    diff[nan_mask] = 0
                    # TODO: implement abs on CharTensor
                    if diff.is_signed() and 'CharTensor' not in diff.type():
                        diff = diff.abs()
                    max_err = diff.max()
                    self.assertLessEqual(max_err, prec, message)

            super(TestCase, self).assertEqual(x.is_sparse, y.is_sparse, message)
            if x.is_sparse:
                x = self.safeCoalesce(x)
                y = self.safeCoalesce(y)
                assertTensorsEqual(x._indices(), y._indices())
                assertTensorsEqual(x._values(), y._values())
            else:
                assertTensorsEqual(x, y)
        elif isinstance(x, string_classes) and isinstance(y, string_classes):
            super(TestCase, self).assertEqual(x, y, message)
        elif type(x) == set and type(y) == set:
            super(TestCase, self).assertEqual(x, y, message)
        elif is_iterable(x) and is_iterable(y):
            super(TestCase, self).assertEqual(len(x), len(y), message)
            for x_, y_ in zip(x, y):
                self.assertEqual(x_, y_, prec, message)
        elif isinstance(x, bool) and isinstance(y, bool):
            super(TestCase, self).assertEqual(x, y, message)
        elif isinstance(x, Number) and isinstance(y, Number):
            if abs(x) == inf or abs(y) == inf:
                if allow_inf:
                    super(TestCase, self).assertEqual(x, y, message)
                else:
                    self.fail("Expected finite numeric values - x={}, y={}".format(x, y))
                return
            super(TestCase, self).assertLessEqual(abs(x - y), prec, message)
        else:
            super(TestCase, self).assertEqual(x, y, message)

    def assertAlmostEqual(self, x, y, places=None, msg=None, delta=None, allow_inf=None):
        prec = delta
        if places:
            prec = 10**(-places)
        self.assertEqual(x, y, prec, msg, allow_inf)

    def assertNotEqual(self, x, y, prec=None, message=''):
        if isinstance(prec, str) and message == '':
            message = prec
            prec = None
        if prec is None:
            prec = self.precision

        if isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor):
            if x.size() != y.size():
                super(TestCase, self).assertNotEqual(x.size(), y.size())
            self.assertGreater(x.numel(), 0)
            y = y.type_as(x)
            y = y.cuda(device=x.get_device()) if x.is_cuda else y.cpu()
            nan_mask = x != x
            if torch.equal(nan_mask, y != y):
                diff = x - y
                if diff.is_signed():
                    diff = diff.abs()
                diff[nan_mask] = 0
                max_err = diff.max()
                self.assertGreaterEqual(max_err, prec, message)
        elif type(x) == str and type(y) == str:
            super(TestCase, self).assertNotEqual(x, y)
        elif is_iterable(x) and is_iterable(y):
            super(TestCase, self).assertNotEqual(x, y)
        else:
            try:
                self.assertGreaterEqual(abs(x - y), prec, message)
                return
            except (TypeError, AssertionError):
                pass
            super(TestCase, self).assertNotEqual(x, y, message)

    def assertObjectIn(self, obj, iterable):
        for elem in iterable:
            if id(obj) == id(elem):
                return
        raise AssertionError("object not found in iterable")

    # TODO: Support context manager interface
    # NB: The kwargs forwarding to callable robs the 'subname' parameter.
    # If you need it, manually apply your callable in a lambda instead.
    def assertExpectedRaises(self, exc_type, callable, *args, **kwargs):
        subname = None
        if 'subname' in kwargs:
            subname = kwargs['subname']
            del kwargs['subname']
        try:
            callable(*args, **kwargs)
        except exc_type as e:
            self.assertExpected(str(e), subname)
            return
        # Don't put this in the try block; the AssertionError will catch it
        self.fail(msg="Did not raise when expected to")

    def assertWarns(self, callable, msg=''):
        r"""
        Test if :attr:`callable` raises a warning.
        """
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter("always")  # allow any warning to be raised
            callable()
            self.assertTrue(len(ws) > 0, msg)

    def assertWarnsRegex(self, callable, regex, msg=''):
        r"""
        Test if :attr:`callable` raises any warning with message that contains
        the regex pattern :attr:`regex`.
        """
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter("always")  # allow any warning to be raised
            callable()
            self.assertTrue(len(ws) > 0, msg)
            found = any(re.search(regex, str(w.message)) is not None for w in ws)
            self.assertTrue(found, msg)

    def assertExpected(self, s, subname=None):
        r"""
        Test that a string matches the recorded contents of a file
        derived from the name of this test and subname.  This file
        is placed in the 'expect' directory in the same directory
        as the test script. You can automatically update the recorded test
        output using --accept.
        If you call this multiple times in a single function, you must
        give a unique subname each time.
        """
        if not (isinstance(s, str) or (sys.version_info[0] == 2 and isinstance(s, unicode))):
            raise TypeError("assertExpected is strings only")

        def remove_prefix(text, prefix):
            if text.startswith(prefix):
                return text[len(prefix):]
            return text

        # NB: we take __file__ from the module that defined the test
        # class, so we place the expect directory where the test script
        # lives, NOT where test/common.py lives.  This doesn't matter in
        # PyTorch where all test scripts are in the same directory as
        # test/common.py, but it matters in onnx-pytorch
        module_id = self.__class__.__module__
        munged_id = remove_prefix(self.id(), module_id + ".")
        test_file = os.path.realpath(sys.modules[module_id].__file__)
        expected_file = os.path.join(os.path.dirname(test_file), "expect", munged_id)
        if subname:
            expected_file += "-" + subname
        expected_file += ".expect"
        expected = None

        def accept_output(update_type):
            print("Accepting {} for {}:\n\n{}".format(update_type, munged_id, s))
            with open(expected_file, 'w') as f:
                f.write(s)

        try:
            with open(expected_file) as f:
                expected = f.read()
        except IOError as e:
            if e.errno != errno.ENOENT:
                raise
            elif ACCEPT:
                return accept_output("output")
            else:
                raise RuntimeError(("I got this output for {}:\n\n{}\n\n"
                                    "No expect file exists; to accept the current output, run:\n"
                                    "python {} {} --accept").format(munged_id, s, __main__.__file__,
                                                                    munged_id))

        # a hack for JIT tests
        if IS_WINDOWS:
            expected = re.sub(r'CppOp\[(.+?)\]', 'CppOp[]', expected)
            s = re.sub(r'CppOp\[(.+?)\]', 'CppOp[]', s)

        if ACCEPT:
            if expected != s:
                return accept_output("updated output")
        else:
            if hasattr(self, "assertMultiLineEqual"):
                # Python 2.7 only
                # NB: Python considers lhs "old" and rhs "new".
                self.assertMultiLineEqual(expected, s)
            else:
                self.assertEqual(s, expected)

    if sys.version_info < (3, 2):
        # assertRegexpMatches renamed to assertRegex in 3.2
        assertRegex = unittest.TestCase.assertRegexpMatches
        # assertRaisesRegexp renamed to assertRaisesRegex in 3.2
        assertRaisesRegex = unittest.TestCase.assertRaisesRegexp


def download_file(url, binary=True):
    if sys.version_info < (3,):
        from urlparse import urlsplit
        import urllib2
        request = urllib2
        error = urllib2
    else:
        from urllib.parse import urlsplit
        from urllib import request, error

    filename = os.path.basename(urlsplit(url)[2])
    data_dir = get_writable_path(os.path.join(os.path.dirname(__file__), 'data'))
    path = os.path.join(data_dir, filename)

    if os.path.exists(path):
        return path
    try:
        data = request.urlopen(url, timeout=15).read()
        with open(path, 'wb' if binary else 'w') as f:
            f.write(data)
        return path
    except error.URLError:
        msg = "could not download test file '{}'".format(url)
        warnings.warn(msg, RuntimeWarning)
        raise unittest.SkipTest(msg)


# We cannot import TEST_CUDA from common_nn here, because if we do that,
# the TEST_CUDNN line from common_nn will be executed multiple times
# as well during the execution of this test suite, and it will cause
# CUDA OOM error on Windows.
TEST_CUDA = torch.cuda.is_available()

# We need spawn start method for test_manager_unclean_exit, but
# Python 2.7 doesn't allow it.
if sys.version_info[0] == 3:
    # Get a multiprocessing context because some test / third party library will
    # set start_method when imported, and setting again triggers RuntimeError.
    mp = mp.get_context(method='spawn')

JOIN_TIMEOUT = 17.0 if IS_WINDOWS else 6.5


class TestDatasetRandomSplit(TestCase):

    def test_lengths_must_equal_datset_size(self):
        with self.assertRaises(ValueError):
            random_split([1, 2, 3, 4], [1, 2])

    def test_splits_have_correct_size(self):
        splits = random_split([1, 2, 3, 4, 5, 6], [2, 4])
        self.assertEqual(len(splits), 2)
        self.assertEqual(len(splits[0]), 2)
        self.assertEqual(len(splits[1]), 4)

    def test_splits_are_mutually_exclusive(self):
        data = [5, 2, 3, 4, 1, 6]
        splits = random_split(data, [2, 4])
        all_values = []
        all_values.extend(list(splits[0]))
        all_values.extend(list(splits[1]))
        data.sort()
        all_values.sort()
        self.assertListEqual(data, all_values)


class TestTensorDataset(TestCase):

    def test_len(self):
        source = TensorDataset(torch.randn(15, 10, 2, 3, 4, 5), torch.randperm(15))
        self.assertEqual(len(source), 15)

    def test_getitem(self):
        t = torch.randn(15, 10, 2, 3, 4, 5)
        l = torch.randn(15, 10)
        source = TensorDataset(t, l)
        for i in range(15):
            self.assertEqual(t[i], source[i][0])
            self.assertEqual(l[i], source[i][1])

    def test_getitem_1d(self):
        t = torch.randn(15)
        l = torch.randn(15)
        source = TensorDataset(t, l)
        for i in range(15):
            self.assertEqual(t[i], source[i][0])
            self.assertEqual(l[i], source[i][1])

    def test_single_tensor(self):
        t = torch.randn(5, 10)
        source = TensorDataset(t)
        self.assertEqual(len(source), 5)
        for i in range(5):
            self.assertEqual(t[i], source[i][0])

    def test_many_tensors(self):
        t0 = torch.randn(5, 10, 2, 3, 4, 5)
        t1 = torch.randn(5, 10)
        t2 = torch.randn(5, 10, 2, 5)
        t3 = torch.randn(5, 10, 3, 7)
        source = TensorDataset(t0, t1, t2, t3)
        self.assertEqual(len(source), 5)
        for i in range(5):
            self.assertEqual(t0[i], source[i][0])
            self.assertEqual(t1[i], source[i][1])
            self.assertEqual(t2[i], source[i][2])
            self.assertEqual(t3[i], source[i][3])


class TestConcatDataset(TestCase):

    def test_concat_two_singletons(self):
        result = ConcatDataset([[0], [1]])
        self.assertEqual(2, len(result))
        self.assertEqual(0, result[0])
        self.assertEqual(1, result[1])

    def test_concat_two_non_singletons(self):
        result = ConcatDataset([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
        self.assertEqual(10, len(result))
        self.assertEqual(0, result[0])
        self.assertEqual(5, result[5])

    def test_concat_two_non_singletons_with_empty(self):
        # Adding an empty dataset somewhere is correctly handled
        result = ConcatDataset([[0, 1, 2, 3, 4], [], [5, 6, 7, 8, 9]])
        self.assertEqual(10, len(result))
        self.assertEqual(0, result[0])
        self.assertEqual(5, result[5])

    def test_concat_raises_index_error(self):
        result = ConcatDataset([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
        with self.assertRaises(IndexError):
            # this one goes to 11
            result[11]

    def test_add_dataset(self):
        d1 = TensorDataset(torch.rand(7, 3, 28, 28), torch.rand(7))
        d2 = TensorDataset(torch.rand(7, 3, 28, 28), torch.rand(7))
        d3 = TensorDataset(torch.rand(7, 3, 28, 28), torch.rand(7))
        result = d1 + d2 + d3
        self.assertEqual(21, len(result))
        self.assertEqual(0, (d1[0][0] - result[0][0]).abs().sum())
        self.assertEqual(0, (d2[0][0] - result[7][0]).abs().sum())
        self.assertEqual(0, (d3[0][0] - result[14][0]).abs().sum())


# Stores the first encountered exception in .exception.
# Inspired by https://stackoverflow.com/a/33599967
class ErrorTrackingProcess(mp.Process):

    def __init__(self, *args, **kwargs):
        super(ErrorTrackingProcess, self).__init__(*args, **kwargs)
        self._pconn, self._cconn = mp.Pipe()
        self._exception = None

    def run(self):
        # Disable stderr printing from os level, and make workers not printing
        # to stderr.
        # Can't use sys.stderr.close, otherwise Python `raise` will error with
        # ValueError: I/O operation on closed file.
        os.close(sys.stderr.fileno())
        try:
            super(ErrorTrackingProcess, self).run()
            self._cconn.send(None)
        except Exception as e:
            self._cconn.send(ExceptionWrapper(sys.exc_info()))
            raise

    @property
    def exception(self):
        if self._pconn.poll():
            self._exception = self._pconn.recv()
        if self._exception is None:
            return None
        else:
            return self._exception.exc_type(self._exception.exc_msg)

    # ESRCH means that os.kill can't finds alive proc
    def send_signal(self, signum, ignore_ESRCH=False):
        try:
            os.kill(self.pid, signum)
        except OSError as e:
            if not ignore_ESRCH or e.errno != errno.ESRCH:
                raise


class ErrorDataset(Dataset):

    def __init__(self, size):
        self.size = size

    def __len__(self):
        return self.size


class SegfaultDataset(Dataset):

    def __init__(self, size):
        self.size = size

    def __getitem__(self, idx):
        return ctypes.string_at(0)

    def __len__(self):
        return self.size


class SleepDataset(Dataset):

    def __init__(self, size, sleep_sec):
        self.size = size
        self.sleep_sec = sleep_sec
        self.sleeped = False

    def __getitem__(self, idx):
        if not self.sleeped:
            time.sleep(self.sleep_sec)
            self.sleeped = True
        return idx

    def __len__(self):
        return self.size


class SeedDataset(Dataset):

    def __init__(self, size):
        self.size = size

    def __getitem__(self, idx):
        return torch.initial_seed()

    def __len__(self):
        return self.size


# Inspired by https://stackoverflow.com/a/26703365
# This will ensure that each worker at least processes one data
class SynchronizedSeedDataset(Dataset):

    def __init__(self, size, num_workers):
        assert size >= num_workers
        self.count = mp.Value('i', 0, lock=True)
        self.barrier = mp.Semaphore(0)
        self.num_workers = num_workers
        self.size = size

    def __getitem__(self, idx):
        with self.count.get_lock():
            self.count.value += 1
            if self.count.value == self.num_workers:
                self.barrier.release()
        self.barrier.acquire()
        self.barrier.release()
        return torch.initial_seed()

    def __len__(self):
        return self.size


def _test_timeout():
    dataset = SleepDataset(10, 3)
    dataloader = DataLoader(dataset, batch_size=2, num_workers=2, timeout=1)
    _ = next(iter(dataloader))


def _test_segfault():
    dataset = SegfaultDataset(10)
    dataloader = DataLoader(dataset, batch_size=2, num_workers=2)
    _ = next(iter(dataloader))


# test custom init function
def init_fn(worker_id):
    torch.manual_seed(12345)


class TestDataLoader(TestCase):

    def setUp(self):
        self.data = torch.randn(100, 2, 3, 5)
        self.labels = torch.randperm(50).repeat(2)
        self.dataset = TensorDataset(self.data, self.labels)

    def _test_sequential(self, loader):
        batch_size = loader.batch_size
        for i, (sample, target) in enumerate(loader):
            idx = i * batch_size
            self.assertEqual(sample, self.data[idx:idx + batch_size])
            self.assertEqual(target, self.labels[idx:idx + batch_size])
        self.assertEqual(i, math.floor((len(self.dataset) - 1) / batch_size))

    def _test_shuffle(self, loader):
        found_data = {i: 0 for i in range(self.data.size(0))}
        found_labels = {i: 0 for i in range(self.labels.size(0))}
        batch_size = loader.batch_size
        for i, (batch_samples, batch_targets) in enumerate(loader):
            for sample, target in zip(batch_samples, batch_targets):
                for data_point_idx, data_point in enumerate(self.data):
                    if data_point.eq(sample).all():
                        self.assertFalse(found_data[data_point_idx])
                        found_data[data_point_idx] += 1
                        break
                self.assertEqual(target, self.labels[data_point_idx])
                found_labels[data_point_idx] += 1
            self.assertEqual(sum(found_data.values()), (i + 1) * batch_size)
            self.assertEqual(sum(found_labels.values()), (i + 1) * batch_size)
        self.assertEqual(i, math.floor((len(self.dataset) - 1) / batch_size))

    def _test_error(self, loader):
        it = iter(loader)
        errors = 0
        while True:
            try:
                next(it)
            except NotImplementedError:
                errors += 1
            except StopIteration:
                self.assertEqual(errors, math.ceil(float(len(loader.dataset)) / loader.batch_size))
                return

    def test_invalid_assign_after_init(self):
        dl = DataLoader(self.dataset)
        for attr in ('batch_size', 'sampler', 'drop_last'):

            def fn():
                setattr(dl, attr, {})

            self.assertRaises(ValueError, fn)

    def test_sequential(self):
        self._test_sequential(DataLoader(self.dataset))

    def test_sequential_batch(self):
        self._test_sequential(DataLoader(self.dataset, batch_size=2))

    def test_growing_dataset(self):
        dataset = [torch.ones(4) for _ in range(4)]
        dataloader_seq = DataLoader(dataset, shuffle=False)
        dataloader_shuffle = DataLoader(dataset, shuffle=True)
        dataset.append(torch.ones(4))
        self.assertEqual(len(dataloader_seq), 5)
        self.assertEqual(len(dataloader_shuffle), 5)

    @unittest.skipIf(not TEST_CUDA, "CUDA unavailable")
    def test_sequential_pin_memory(self):
        loader = DataLoader(self.dataset, batch_size=2, pin_memory=True)
        for input, target in loader:
            self.assertTrue(input.is_pinned())
            self.assertTrue(target.is_pinned())

    def test_multiple_dataloaders(self):
        loader1_it = iter(DataLoader(self.dataset, num_workers=1))
        loader2_it = iter(DataLoader(self.dataset, num_workers=2))
        next(loader1_it)
        next(loader1_it)
        next(loader2_it)
        next(loader2_it)
        next(loader1_it)
        next(loader2_it)

    @unittest.skip("temporarily disable until flaky failures are fixed")
    def test_segfault(self):
        p = ErrorTrackingProcess(target=_test_segfault)
        p.start()
        p.join(JOIN_TIMEOUT)
        try:
            self.assertFalse(p.is_alive())
            self.assertNotEqual(p.exitcode, 0)
            if IS_WINDOWS:
                self.assertIsInstance(p.exception, OSError)
                self.assertRegex(str(p.exception), r'access violation reading ')
            else:
                self.assertIsInstance(p.exception, RuntimeError)
                self.assertRegex(
                    str(p.exception), r'DataLoader worker \(pid \d+\) is killed by signal: ')
        finally:
            p.terminate()

    def test_timeout(self):
        p = ErrorTrackingProcess(target=_test_timeout)
        p.start()
        p.join(JOIN_TIMEOUT)
        try:
            self.assertFalse(p.is_alive())
            self.assertNotEqual(p.exitcode, 0)
            self.assertIsInstance(p.exception, RuntimeError)
            self.assertRegex(str(p.exception), r'DataLoader timed out after \d+ seconds')
        finally:
            p.terminate()

    def test_worker_seed(self):
        num_workers = 6
        dataset = SynchronizedSeedDataset(num_workers, num_workers)
        dataloader = DataLoader(dataset, batch_size=1, num_workers=num_workers)
        seeds = set()
        for batch in dataloader:
            seeds.add(batch[0])
        self.assertEqual(len(seeds), num_workers)

    def test_worker_init_fn(self):
        dataset = SeedDataset(4)
        dataloader = DataLoader(dataset, batch_size=2, num_workers=2, worker_init_fn=init_fn)
        for batch in dataloader:
            self.assertEqual(12345, batch[0])
            self.assertEqual(12345, batch[1])

    def test_shuffle(self):
        self._test_shuffle(DataLoader(self.dataset, shuffle=True))

    def test_shuffle_batch(self):
        self._test_shuffle(DataLoader(self.dataset, batch_size=2, shuffle=True))

    def test_sequential_workers(self):
        self._test_sequential(DataLoader(self.dataset, num_workers=4))

    def test_seqential_batch_workers(self):
        self._test_sequential(DataLoader(self.dataset, batch_size=2, num_workers=4))

    def test_shuffle_workers(self):
        self._test_shuffle(DataLoader(self.dataset, shuffle=True, num_workers=4))

    def test_shuffle_batch_workers(self):
        self._test_shuffle(DataLoader(self.dataset, batch_size=2, shuffle=True, num_workers=4))

    def _test_batch_sampler(self, **kwargs):
        # [(0, 1), (2, 3, 4), (5, 6), (7, 8, 9), ...]
        batches = []
        for i in range(0, 100, 5):
            batches.append(tuple(range(i, i + 2)))
            batches.append(tuple(range(i + 2, i + 5)))

        dl = DataLoader(self.dataset, batch_sampler=batches, **kwargs)
        self.assertEqual(len(dl), 40)
        for i, (input, _target) in enumerate(dl):
            if i % 2 == 0:
                offset = i * 5 // 2
                self.assertEqual(len(input), 2)
                self.assertEqual(input, self.data[offset:offset + 2])
            else:
                offset = i * 5 // 2
                self.assertEqual(len(input), 3)
                self.assertEqual(input, self.data[offset:offset + 3])

    @unittest.skipIf(NO_MULTIPROCESSING_SPAWN, "Disabled for environments that \
                     don't support multiprocessing with spawn start method")
    def test_batch_sampler(self):
        self._test_batch_sampler()
        self._test_batch_sampler(num_workers=4)

    @unittest.skipIf(not TEST_CUDA, "CUDA unavailable")
    def test_shuffle_pin_memory(self):
        loader = DataLoader(
            self.dataset, batch_size=2, shuffle=True, num_workers=4, pin_memory=True)
        for input, target in loader:
            self.assertTrue(input.is_pinned())
            self.assertTrue(target.is_pinned())

    @unittest.skipIf(not TEST_NUMPY, "numpy unavailable")
    def test_numpy(self):
        import numpy as np

        class TestDataset(torch.utils.data.Dataset):

            def __getitem__(self, i):
                return np.ones((2, 3, 4)) * i

            def __len__(self):
                return 1000

        loader = DataLoader(TestDataset(), batch_size=12)
        batch = next(iter(loader))
        self.assertIsInstance(batch, torch.DoubleTensor)
        self.assertEqual(batch.size(), torch.Size([12, 2, 3, 4]))

    def test_error(self):
        self._test_error(DataLoader(ErrorDataset(100), batch_size=2, shuffle=True))

    @unittest.skipIf(NO_MULTIPROCESSING_SPAWN, "Disabled for environments that \
                     don't support multiprocessing with spawn start method")
    def test_error_workers(self):
        self._test_error(DataLoader(ErrorDataset(41), batch_size=2, shuffle=True, num_workers=4))

    @unittest.skipIf(IS_WINDOWS, "FIXME: stuck test")
    @unittest.skipIf(not TEST_CUDA, "CUDA unavailable")
    def test_partial_workers(self):
        r"""Check that workers exit even if the iterator is not exhausted."""
        for pin_memory in (True, False):
            loader = iter(
                DataLoader(self.dataset, batch_size=2, num_workers=4, pin_memory=pin_memory))
            workers = loader.workers
            if pin_memory:
                pin_memory_thread = loader.pin_memory_thread
            for i, sample in enumerate(loader):
                if i == 10:
                    break
            del loader
            for w in workers:
                w.join(JOIN_TIMEOUT)
                self.assertFalse(w.is_alive(), 'subprocess not terminated')
            if pin_memory:
                pin_memory_thread.join(JOIN_TIMEOUT)
                self.assertFalse(pin_memory_thread.is_alive())

    @staticmethod
    def _main_process(dataset, worker_pids, main_exit_event, raise_error):
        loader = iter(DataLoader(dataset, batch_size=2, num_workers=4, pin_memory=True))
        workers = loader.workers
        for i in range(len(workers)):
            worker_pids[i] = int(workers[i].pid)
        for i, sample in enumerate(loader):
            if i == 3:
                # Simulate an exit of the manager process
                main_exit_event.set()
                if raise_error:
                    raise RuntimeError('Error')
                else:
                    if IS_WINDOWS:
                        os.system('taskkill /PID ' + str(os.getpid()) + ' /F')
                    else:
                        os.kill(os.getpid(), signal.SIGKILL)

    @staticmethod
    def _is_process_alive(pid, pname):
        # There is a chance of a terminated child process's pid being reused by a new unrelated
        # process, but since we are looping this check very frequently, we will know that the child
        # process dies before the new unrelated process starts.
        if IS_WINDOWS:
            command = 'tasklist | find "{}" /i'.format(pid)
        else:
            command = 'ps -p {} -o comm='.format(pid)
        p = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
        (output, err) = p.communicate()
        p_status = p.wait()
        output = output.decode('utf-8')
        return pname in output

    @unittest.skipIf(NO_MULTIPROCESSING_SPAWN, "Disabled for environments that \
                     don't support multiprocessing with spawn start method")
    @unittest.skipIf(sys.version_info[0] == 2, "spawn start method is not supported in Python 2, \
                     but we need it for creating another process with CUDA")
    @unittest.skipIf(not TEST_CUDA, "CUDA unavailable")
    @skipIfRocm
    def test_main_process_unclean_exit(self):
        r'''There might be ConnectionResetError or leaked semaphore warning (due to dirty process exit), \
but they are all safe to ignore'''

        # `raise_error` controls if the main process is KILL-ed by OS or just
        # simply raises an error. Both cases are interesting because
        # 1. In case of it is KILL-ed by OS, the workers need to automatically
        #    discover that their parent is dead and exit gracefully.
        # 2. In case of it raises an error itself, the parent process needs to
        #    take care of exiting the worker and then exits itself gracefully.
        for raise_error in (True, False):
            worker_pids = mp.Array('i', [0] * 4)

            main_exit_event = mp.Event()
            p = mp.Process(
                target=TestDataLoader._main_process,
                args=(self.dataset, worker_pids, main_exit_event, raise_error))
            p.start()
            worker_pids[-1] = p.pid

            main_exit_event.wait()

            exit_status = [False] * len(worker_pids)
            start_time = time.time()
            pname = 'python'
            while True:
                for i in range(len(worker_pids)):
                    pid = worker_pids[i]
                    if not exit_status[i]:
                        if not TestDataLoader._is_process_alive(pid, pname):
                            exit_status[i] = True
                if all(exit_status):
                    break
                else:
                    if time.time() - start_time > MANAGER_STATUS_CHECK_INTERVAL + JOIN_TIMEOUT:
                        self.fail('subprocess not terminated')
                    time.sleep(1)
            p.join(MANAGER_STATUS_CHECK_INTERVAL + JOIN_TIMEOUT - (time.time() - start_time))
            self.assertFalse(p.is_alive(), 'main process not terminated')

    def test_len(self):

        def check_len(dl, expected):
            self.assertEqual(len(dl), expected)
            n = 0
            for sample in dl:
                n += 1
            self.assertEqual(n, expected)

        check_len(self.dataset, 100)
        check_len(DataLoader(self.dataset, batch_size=2), 50)
        check_len(DataLoader(self.dataset, batch_size=3), 34)

    @unittest.skipIf(not TEST_NUMPY, "numpy unavailable")
    def test_numpy_scalars(self):
        import numpy as np

        class ScalarDataset(torch.utils.data.Dataset):

            def __init__(self, dtype):
                self.dtype = dtype

            def __getitem__(self, i):
                return self.dtype()

            def __len__(self):
                return 4

        dtypes = {
            np.float64: torch.DoubleTensor,
            np.float32: torch.FloatTensor,
            np.float16: torch.HalfTensor,
            np.int64: torch.LongTensor,
            np.int32: torch.IntTensor,
            np.int16: torch.ShortTensor,
            np.int8: torch.CharTensor,
            np.uint8: torch.ByteTensor,
        }
        for dt, tt in dtypes.items():
            dset = ScalarDataset(dt)
            loader = DataLoader(dset, batch_size=2)
            batch = next(iter(loader))
            self.assertIsInstance(batch, tt)

    @unittest.skipIf(not TEST_NUMPY, "numpy unavailable")
    def test_default_collate_bad_numpy_types(self):
        import numpy as np

        # Should be a no-op
        arr = np.array(['a', 'b', 'c'])
        default_collate(arr)

        arr = np.array([[['a', 'b', 'c']]])
        self.assertRaises(TypeError, lambda: default_collate(arr))

        arr = np.array([object(), object(), object()])
        self.assertRaises(TypeError, lambda: default_collate(arr))

        arr = np.array([[[object(), object(), object()]]])
        self.assertRaises(TypeError, lambda: default_collate(arr))


class StringDataset(Dataset):

    def __init__(self):
        self.s = '12345'

    def __len__(self):
        return len(self.s)

    def __getitem__(self, ndx):
        return (self.s[ndx], ndx)


class TestStringDataLoader(TestCase):

    def setUp(self):
        self.dataset = StringDataset()

    @unittest.skipIf(not TEST_CUDA, "CUDA unavailable")
    def test_shuffle_pin_memory(self):
        loader = DataLoader(
            self.dataset, batch_size=2, shuffle=True, num_workers=4, pin_memory=True)
        for batch_ndx, (s, n) in enumerate(loader):
            self.assertIsInstance(s[0], str)
            self.assertTrue(n.is_pinned())


class DictDataset(Dataset):

    def __len__(self):
        return 4

    def __getitem__(self, ndx):
        return {
            'a_tensor': torch.Tensor(4, 2).fill_(ndx),
            'another_dict': {
                'a_number': ndx,
            },
        }


class TestDictDataLoader(TestCase):

    def setUp(self):
        self.dataset = DictDataset()

    def test_sequential_batch(self):
        loader = DataLoader(self.dataset, batch_size=2, shuffle=False)
        batch_size = loader.batch_size
        for i, sample in enumerate(loader):
            idx = i * batch_size
            self.assertEqual(set(sample.keys()), {'a_tensor', 'another_dict'})
            self.assertEqual(set(sample['another_dict'].keys()), {'a_number'})

            t = sample['a_tensor']
            self.assertEqual(t.size(), torch.Size([batch_size, 4, 2]))
            self.assertTrue((t[0] == idx).all())
            self.assertTrue((t[1] == idx + 1).all())

            n = sample['another_dict']['a_number']
            self.assertEqual(n.size(), torch.Size([batch_size]))
            self.assertEqual(n[0], idx)
            self.assertEqual(n[1], idx + 1)

    @unittest.skipIf(not TEST_CUDA, "CUDA unavailable")
    def test_pin_memory(self):
        loader = DataLoader(self.dataset, batch_size=2, pin_memory=True)
        for batch_ndx, sample in enumerate(loader):
            self.assertTrue(sample['a_tensor'].is_pinned())
            self.assertTrue(sample['another_dict']['a_number'].is_pinned())


class _TestWorkerQueueDataset(Dataset):

    def __init__(self, data):
        self.data = data
        self.worker_id = None

    def worker_init_fn(self, worker_id):
        self.worker_id = worker_id

    def __getitem__(self, item):
        return self.worker_id, self.data[item]

    def __len__(self):
        return len(self.data)


class TestIndividualWorkerQueue(TestCase):

    def setUp(self):
        self.dataset = _TestWorkerQueueDataset([i for i in range(128)])

    def _run_ind_worker_queue_test(self, batch_size, num_workers):
        loader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            worker_init_fn=self.dataset.worker_init_fn)
        current_worker_idx = 0
        for i, (worker_ids, sample) in enumerate(loader):
            self.assertEqual(worker_ids.tolist(), [current_worker_idx] * batch_size)
            self.assertEqual(sample.tolist(),
                             [j for j in range(i * batch_size, (i + 1) * batch_size)])
            current_worker_idx += 1
            if current_worker_idx == num_workers:
                current_worker_idx = 0

    def test_ind_worker_queue(self):
        for batch_size in (8, 16, 32, 64):
            for num_workers in range(1, 6):
                self._run_ind_worker_queue_test(batch_size=batch_size, num_workers=num_workers)


if __name__ == '__main__':
    run_tests()

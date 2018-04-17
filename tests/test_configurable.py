import unittest
import numpy as np

from src.configurable import _dict_to_flat_config
from src.configurable import add_config
from src.configurable import clear_config
from src.configurable import configurable
from src.configurable import log_config


@configurable
def mock_func(*args, **kwargs):
    return args, kwargs


@configurable
def mock_func_2(arg, kwarg=None):
    return arg, kwarg


class MockClass(object):

    @configurable
    def __init__(self, arg):
        self.arg = arg

    @configurable
    def mock_func(self, arg):
        return arg


class MockClass2(object):

    def __init__(self, arg):
        self.arg = arg


class TestConfigurable(unittest.TestCase):

    def setUp(self):
        self.defaults = {
            'mock_func': {
                'arg': 'arg'
            },
            'mock_func_2.kwarg': 'kwarg',
            'mock_func_2': {
                'arg': 'arg',
            },
            'MockClass': {
                '__init__': {
                    'arg': 'arg'
                },
                'mock_func': {
                    'arg': 'arg'
                }
            },
            'MockClass2': {
                '__init__': {
                    'arg': 'arg'
                }
            }
        }

    def tearDown(self):
        clear_config()

    def test_dict_to_flat_config(self):
        dict_ = {
            'mock_func_2.kwarg': 'kwarg',
            'mock_func_2': {
                'arg': 'arg',
            },
        }
        expected = {'mock_func_2.kwarg': 'kwarg', 'mock_func_2.arg': 'arg'}
        flat_dict = _dict_to_flat_config(dict_)
        self.assertEqual(flat_dict, expected)

    def test_dict_to_flat_config_invalid(self):
        dict_ = {
            'mock_func_2.kwarg': 'kwarg',
            'mock_func_2': {
                'arg': 'arg',
            },
            'mock_func_2.arg': 'arg',
        }
        self.assertRaises(TypeError, lambda: _dict_to_flat_config(dict_))

    def test_add_config_numpy(self):
        # Regression test, add_config printer failed with numpy types
        self.defaults['mock_func.kwarg'] = np.float32(1.0)
        add_config({__name__: self.defaults})

    def test_invalid_duplicate_key(self):
        self.defaults['mock_func.arg'] = 'arg_duplicate'
        self.assertRaises(TypeError, lambda: add_config({__name__: self.defaults}))

    def test_invalid_invalid_key(self):
        self.defaults['mock_func.arg.'] = 'arg'
        self.assertRaises(TypeError, lambda: add_config({__name__: self.defaults}))

    def test_invalid_invalid_key_2(self):
        self.defaults['.'] = 'arg'
        self.assertRaises(TypeError, lambda: add_config({__name__: self.defaults}))

    def test_invalid_invalid_key_3(self):
        self.defaults['.mock_func'] = 'arg'
        self.assertRaises(TypeError, lambda: add_config({__name__: self.defaults}))

    def test_invalid_invalid_key_4(self):
        self.defaults['mock_func..arg'] = 'arg'
        self.assertRaises(TypeError, lambda: add_config({__name__: self.defaults}))

    def test_merge_config_dict(self):
        add_config({__name__: self.defaults})
        add_config({__name__: {'mock_func_2': {'arg': 'arg_new'}}})
        self.assertEqual(('arg_new', 'kwarg'), mock_func_2())

    def test_merge_config_dots(self):
        add_config({__name__: self.defaults})
        add_config({__name__: {'mock_func_2.kwarg': 'kwarg_new'}})
        self.assertEqual(('arg', 'kwarg_new'), mock_func_2())

    def test_mock_func(self):
        add_config({__name__: self.defaults})
        self.assertEqual(self.defaults['mock_func'], mock_func()[1])

    def test_mock_func_clean(self):
        add_config({__name__: self.defaults})
        clear_config()
        self.assertNotEqual(self.defaults['mock_func'], mock_func()[1])

    def test_mock_func_var_args(self):
        add_config({__name__: self.defaults})
        self.assertEqual(mock_func('arg')[0][0], 'arg')

    def test_mock_func_2(self):
        add_config({__name__: self.defaults})
        self.assertEqual(('arg', 'kwarg'), mock_func_2())

    def test_mock_func_2_log_config(self):
        """ Ensure log_config does not break """
        add_config({__name__: self.defaults})
        log_config()

    def test_mock_func_2_missing_argument(self):
        del self.defaults['mock_func_2']['arg']
        add_config(self.defaults)
        self.assertRaises(TypeError, lambda: mock_func_2())

    def test_mock_func_2_extra_kwarg(self):
        self.defaults['mock_func_2']['kwarg_extra'] = 'b'
        add_config(self.defaults)
        self.assertRaises(TypeError, lambda: mock_func_2())

    def test_mock_func_2_override(self):
        add_config({__name__: self.defaults})
        defaults_mock_func_2 = tuple(self.defaults['mock_func_2'].values())
        self.assertNotEqual(defaults_mock_func_2, mock_func_2('other_arg'))

    def test_mock_class_init(self):
        add_config({__name__: self.defaults})
        self.assertEqual(MockClass().arg, self.defaults['MockClass']['__init__']['arg'])

    def test_mock_class_func(self):
        add_config({__name__: self.defaults})
        self.assertEqual(MockClass().mock_func(), self.defaults['MockClass']['mock_func']['arg'])

    def test_decorate_mock_class(self):
        add_config({__name__: self.defaults})
        MockClass2.__init__ = configurable(MockClass2.__init__)
        self.assertEqual(MockClass2().arg, self.defaults['MockClass2']['__init__']['arg'])

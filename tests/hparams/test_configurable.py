import pytest
import _pytest
import inspect
from unittest import mock

from src.hparams.configurable_ import _check_configuration
from src.hparams.configurable_ import _merge_args
from src.hparams.configurable_ import _parse_configuration
from src.hparams.configurable_ import add_config
from src.hparams.configurable_ import clear_config
from src.hparams.configurable_ import configurable


def test_parse_configuration_example():
    # Test a simple case
    parsed = _parse_configuration({'abc.abc': {'cda': 'abc'}})
    assert parsed == {'abc': {'abc': {'cda': 'abc'}}}


def test_parse_configuration_improper_format():
    # Test if the key is improperly formatted, TypeError is raised
    with pytest.raises(TypeError):
        _parse_configuration({'abc..abc': 'abc'})


def test_parse_configuration_improper_format_2():
    # Test if the key is improperly formatted, TypeError is raised
    with pytest.raises(TypeError):
        _parse_configuration({'abc.abc.': 'abc'})


def test_parse_configuration_improper_format_3():
    # Test if the key is improperly formatted, TypeError is raised
    with pytest.raises(TypeError):
        _parse_configuration({'.abc.abc': 'abc'})


def test_parse_configuration_improper_format_4():
    # Test if the key is improperly formatted, TypeError is raised
    with pytest.raises(TypeError):
        _parse_configuration({'.': 'abc'})


def test_parse_configuration_duplicate_key():
    # Test if the key is duplicated, TypeError is raised
    with pytest.raises(TypeError):
        _parse_configuration({'abc.abc': 'abc', 'abc': {'abc': 'xyz'}})


@configurable
def mock_configurable(*args, **kwargs):
    # Mock function with configurable
    return kwargs


@configurable
def mock_configurable_limited_args(arg, **kwargs):
    # Mock function with configurable
    return kwargs


def mock_without_configurable(**kwargs):
    # Mock function without configurable
    return kwargs


def test_mock_attributes():
    # Test the attributes mock is give, if it's ``@configurable``
    assert hasattr(mock_configurable, '_configurable')
    assert not hasattr(mock_without_configurable, '_configurable')


_pytest.python_api.approx = configurable(_pytest.python_api.approx)


class Mock(object):

    def __init__(self):
        pass


class MockConfigurable(object):

    @configurable
    def __init__(self):
        pass


def test_mock_configurable_limited_args():
    # Check if TypeError on too many args
    with pytest.raises(TypeError):
        mock_configurable_limited_args('abc', 'abc')


def test_check_configuration_external_libraries():
    # Test that check configuration can check ``configurable`` on external libraries
    _check_configuration({'_pytest': {'python_api': {'approx': {'rel': None}}}})


def test_check_configuration_internal_libraries():
    # Test that check configuration can check ``configurable`` on internal libraries
    _check_configuration({
        'tests': {
            'hparams': {
                'test_configurable': {
                    'mock_configurable': {
                        'kwarg': None
                    }
                }
            }
        }
    })


def test_check_configuration_error_internal_libraries():
    # Test that check configuration fails on internal libraries
    with pytest.raises(TypeError):
        _check_configuration({
            'tests': {
                'hparams': {
                    'test_configurable': {
                        'mock': {
                            'kwarg': None
                        }
                    }
                }
            }
        })


def test_check_configuration_error_external_libraries():
    # Test that check configuration fails on internal libraries
    with pytest.raises(TypeError):
        _check_configuration({'random': {'seed': {'a': 1}}})


def test_check_configuration_class():
    # Test that check configuration works for classes
    _check_configuration({
        'tests': {
            'hparams': {
                'test_configurable': {
                    'MockConfigurable': {
                        '__init__': {
                            'kwarg': None
                        }
                    }
                }
            }
        }
    })


def test_check_configuration_error_class():
    # Test that check configuration works for classes
    with pytest.raises(TypeError):
        _check_configuration({
            'tests': {
                'hparams': {
                    'test_configurable': {
                        'Mock': {
                            '__init__': {
                                'kwarg': None
                            }
                        }
                    }
                }
            }
        })


def test_add_config_and_arguments():
    # Check that a function can be configured
    kwargs = {'xyz': 'xyz'}
    add_config({'tests.hparams.test_configurable.mock_configurable': kwargs})
    assert mock_configurable() == kwargs

    # Reset
    clear_config()

    # Check reset worked
    assert mock_configurable() == {}


@mock.patch('src.hparams.configurable_.logger')
def test_merge_arg_kwarg(logger_mock):
    arg_kwarg = lambda a, b='abc': (a, b)
    parameters = list(inspect.signature(arg_kwarg).parameters.values())

    # Prefer ``args`` over ``other_kwargs``
    merged = _merge_args(parameters, ['a', 'abc'], {}, {'b': 'xyz'})
    assert merged == (['a', 'abc'], {})
    logger_mock.warning.assert_called_once()
    logger_mock.reset_mock()

    # Prefer ``kwargs`` over ``other_kwargs``
    merged = _merge_args(parameters, ['a'], {'b': 'abc'}, {'b': 'xyz'})
    assert merged == (['a'], {'b': 'abc'})
    logger_mock.warning.assert_called_once()
    logger_mock.reset_mock()

    # Prefer ``other_kwargs`` over default argument
    merged = _merge_args(parameters, ['a'], {}, {'b': 'xyz'})
    assert merged == (['a'], {'b': 'xyz'})
    logger_mock.warning.assert_not_called()


@mock.patch('src.hparams.configurable_.logger')
def test_merge_arg_variable(logger_mock):
    """
    For arguments, order matters; therefore, unless we are able to abstract everything into a
    key word argument, we have to keep the ``args`` the same.

    The case where we are unable to shift everything to ``args`` is when there exists a ``*args``.

    For example (a, b) cannot be flipped with kwarg:
    >>> arg_kwarg = lambda a, b='abc': (a, b)
    >>> arg_kwarg('b', a='a')
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
    TypeError: <lambda>() got multiple values for argument 'a'
    """
    arg_kwarg = lambda a, *args, b='abc': (a, args, b)
    parameters = list(inspect.signature(arg_kwarg).parameters.values())

    # Handling of variable ``*args``
    merged = _merge_args(parameters, ['a', 'b', 'c'], {}, {'b': 'xyz'})
    assert merged == (['a', 'b', 'c'], {'b': 'xyz'})
    logger_mock.warning.assert_not_called()
    logger_mock.reset_mock()

    # Handling of variable ``*args``
    merged = _merge_args(parameters, ['a', 'b', 'c'], {}, {'a': 'xyz'})
    assert merged == (['a', 'b', 'c'], {})
    logger_mock.warning.assert_called_once()
    logger_mock.reset_mock()


@mock.patch('src.hparams.configurable_.logger')
def test_merge_kwarg_variable(logger_mock):
    """
    If there exists a ``**kwargs``, then
    """
    arg_kwarg = lambda a, b, **kwargs: (a, b, kwargs)
    parameters = list(inspect.signature(arg_kwarg).parameters.values())

    # Handling of variable ``**kwargs``
    merged = _merge_args(parameters, ['a', 'b'], {}, {'b': 'xyz'})
    assert merged == (['a', 'b'], {})
    logger_mock.warning.assert_called_once()
    logger_mock.reset_mock()

    # Handling of variable ``**kwargs``
    merged = _merge_args(parameters, ['a'], {}, {'b': 'xyz'})
    assert merged == (['a'], {'b': 'xyz'})
    logger_mock.warning.assert_not_called()
    logger_mock.reset_mock()

    # Handling of variable ``**kwargs``
    merged = _merge_args(parameters, ['a'], {}, {'b': 'xyz', 'c': 'abc'})
    assert merged == (['a'], {'b': 'xyz', 'c': 'abc'})
    logger_mock.warning.assert_not_called()
    logger_mock.reset_mock()

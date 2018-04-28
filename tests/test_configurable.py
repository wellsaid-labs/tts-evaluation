import pytest
import inspect

from src.configurable import _parse_configuration
from src.configurable import _check_configuration
from src.configurable import configurable
from src.configurable import add_config
from src.configurable import clear_config
from src.configurable import clear_arguments
from src.configurable import _get_arguments
from src.configurable import _merge_args
from src.configurable import log_arguments


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


def mock(**kwargs):
    # Mock function without configurable
    return kwargs


def test_mock_attributes():
    # Test the attributes mock is give, if it's ``@configurable``
    assert hasattr(mock_configurable, '_configurable')
    assert not hasattr(mock, '_configurable')


pytest.approx = configurable(pytest.approx)


class Mock(object):

    @configurable
    def __init__(self):
        pass


def test_check_configuration_external_libraries():
    # Test that check configuration can check ``configurable`` on external libraries
    _check_configuration({'pytest': {'approx': {'rel': None}}})


def test_check_configuration_internal_libraries():
    # Test that check configuration can check ``configurable`` on internal libraries
    _check_configuration({'tests': {'test_configurable': {'mock_configurable': {'kwarg': None}}}})


def test_check_configuration_error_internal_libraries():
    # Test that check configuration fails on internal libraries
    with pytest.raises(TypeError):
        _check_configuration({'tests': {'test_configurable': {'mock': {'kwarg': None}}}})


def test_check_configuration_error_external_libraries():
    # Test that check configuration fails on internal libraries
    with pytest.raises(TypeError):
        _check_configuration({'random': {'seed': {'a': 1}}})


def test_check_configuration_class():
    # Test that check configuration works for classes
    _check_configuration({'tests': {'test_configurable': {'Mock': {'__init__': {'kwarg': None}}}}})


def test_add_config_and_arguments():
    # Check that a function can be configured
    kwargs = {'xyz': 'xyz'}
    add_config({'tests.test_configurable.mock_configurable': kwargs})
    assert mock_configurable() == kwargs

    # Check that the parameters were recorded
    assert _get_arguments()['tests']['test_configurable']['mock_configurable']['xyz'] == 'xyz'

    # Reset
    clear_config()
    clear_arguments()

    # Check reset worked
    assert mock_configurable() == {}


def test_arguments():
    # Check that the parameters were recorded
    mock_configurable(abc='abc')
    assert _get_arguments()['tests']['test_configurable']['mock_configurable']['abc'] == 'abc'

    # Smoke test for log
    log_arguments()

    clear_arguments()


def test_arguments_many():
    # Check that the parameters were recorded
    arg_kwarg = configurable(lambda a, *args, **kwargs: (a, args, kwargs))
    arg_kwarg('abc', 'def', 'ghi', 'klm', abc='abc')
    arg_kwarg('abc', abc='xyz')
    arg_kwarg('abc', abc='cdf')
    arg_kwarg('abc', abc='ghf')
    arg_kwarg('abc', xyz='abc')
    assert str(_get_arguments()['tests']['test_configurable']['test_arguments_many']['<locals>'][
        '<lambda>']['abc']) == str(['xyz', 'cdf', 'ghf'])
    assert _get_arguments()['tests']['test_configurable']['test_arguments_many']['<locals>'][
        '<lambda>']['xyz'] == 'abc'
    assert _get_arguments()['tests']['test_configurable']['test_arguments_many']['<locals>'][
        '<lambda>']['args'] == ('def', 'ghi', 'klm')
    clear_arguments()


def test_merge_arg_kwarg():
    arg_kwarg = lambda a, b='abc': (a, b)
    parameters = list(inspect.signature(arg_kwarg).parameters.values())

    # Prefer ``args`` over ``other_kwargs``
    merged = _merge_args(parameters, ['a', 'abc'], {}, {'b': 'xyz'})
    assert merged == (['a', 'abc'], {})

    # Prefer ``kwargs`` over ``other_kwargs``
    merged = _merge_args(parameters, ['a'], {'b': 'abc'}, {'b': 'xyz'})
    assert merged == (['a'], {'b': 'abc'})

    # Prefer ``other_kwargs`` over default argument
    merged = _merge_args(parameters, ['a'], {}, {'b': 'xyz'})
    assert merged == (['a'], {'b': 'xyz'})


def test_merge_arg_variable():
    """
    For arguments, order matters; therefore, unless we are able to abstract everything into a
    key word argument, we have to keep the ``args`` the same.

    The case where we are unable to shift everything to ``args`` is when there exists a ``*args``.
    Because some

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

    # Handling of variable ``*args``
    merged = _merge_args(parameters, ['a', 'b', 'c'], {}, {'a': 'xyz'})
    assert merged == (['a', 'b', 'c'], {})


def test_merge_kwarg_variable():
    """
    If there exists a ``**kwargs``, then
    """
    arg_kwarg = lambda a, b, **kwargs: (a, b, kwargs)
    parameters = list(inspect.signature(arg_kwarg).parameters.values())

    # Handling of variable ``**kwargs``
    merged = _merge_args(parameters, ['a', 'b'], {}, {'b': 'xyz'})
    assert merged == (['a', 'b'], {})

    # Handling of variable ``**kwargs``
    merged = _merge_args(parameters, ['a'], {}, {'b': 'xyz'})
    assert merged == (['a'], {'b': 'xyz'})

    # Handling of variable ``**kwargs``
    merged = _merge_args(parameters, ['a'], {}, {'b': 'xyz', 'c': 'abc'})
    assert merged == (['a'], {'b': 'xyz', 'c': 'abc'})

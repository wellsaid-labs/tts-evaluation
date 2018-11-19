from pathlib import Path
from functools import reduce
from functools import wraps
from functools import lru_cache
from importlib import import_module

import inspect
import logging
import operator
import pprint
import sys

pretty_printer = pprint.PrettyPrinter(indent=4)
logger = logging.getLogger(__name__)

# TODO: Add 'No Config' mode preventing runtime warnings.
# TODO: Print a warning if a hparam is being overridden, with a flag to turn the warning off.


class _KeyListDictionary(dict):
    """
    Allows for lists of keys to query a deep dictionary.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, key):
        """ Similar to dict.__getitem__ but allows key to be a list of keys """
        if isinstance(key, list):
            return reduce(operator.getitem, key, self)

        return super().__getitem__(key)

    def __contains__(self, key):
        """ Similar to dict.__contains__ but allows key to be a list of keys """
        if isinstance(key, list):
            pointer = self
            for k in key:
                if k in pointer:
                    pointer = pointer[k]
                else:
                    return False
            return True

        return super().__contains__(key)


_configuration = _KeyListDictionary()  # Global private configuration


def _dict_merge(dict_, merge_dict, overwrite=False):
    """ Recursive `dict` merge.

    `dict_merge` recurses down into dicts nested to an arbitrary depth, updating keys. The
    `merge_dict` is merged into `dict_`.

    Args:
        dict_ (dict): dict onto which the merge is executed
        merge_dict (dict): dict merged into ``dict_``
        overwrite (bool): If True, ``merge_dict`` may overwrite ``dict_`` values.
    """
    for key in merge_dict:
        if key in dict_ and isinstance(merge_dict[key], dict) and isinstance(dict_[key], dict):
            _dict_merge(dict_[key], merge_dict[key], overwrite=overwrite)
        elif key in dict_:
            if isinstance(overwrite, bool) and overwrite:
                dict_[key] = merge_dict[key]
        elif key not in dict_:
            dict_[key] = merge_dict[key]


def _parse_configuration(dict_):
    """ Parses ``dict_`` such that dotted key names are interpreted as multiple keys.

    This configuration parser is intended to replicate python's dotted module names.

    Args:
        dict_ (dict): Dotted dictionary to parse

    Returns:
        (dict): Parsed dictionary.

    Example:
        >>> dict = {
        ...   'abc.abc': {
        ...     'cda': 'abc'
        ...   }
        ... }
        >>> _parse_configuration(dict)
        {'abc': {'abc': {'cda': 'abc'}}}
    """
    parsed = {}
    _parse_configuration_helper(dict_, parsed)
    return parsed


def _parse_configuration_helper(dict_, new_dict):
    """ Recursive helper to ``_parse_configuration``

    Args:
        dict_ (dict): Dotted dictionary to parse.
        new_dict (dict): Parsed dictionary that is created.
    """
    if not isinstance(dict_, dict):
        return

    for key in dict_:
        split = key.split('.')
        past_dict = new_dict
        for i, split_key in enumerate(split):
            if split_key == '':
                raise TypeError('Invalid config: Improper key format %s' % key)
            if i == len(split) - 1 and not isinstance(dict_[key], dict):
                if split_key in new_dict:
                    raise TypeError('Invalid config: Key %s already seen.' % key)
                new_dict[split_key] = dict_[key]
            else:
                if split_key not in new_dict:
                    new_dict[split_key] = {}
                new_dict = new_dict[split_key]
        _parse_configuration_helper(dict_[key], new_dict)
        new_dict = past_dict  # Reset dict


def _check_configuration(dict_):
    """ Check that the configuration ``dict_`` is valid.

    Args:
        dict_ (dict): Parsed dict to check

    Raises:
        (TypeError): If ``dict_`` does not refer to a configurable function.
    """
    return _check_configuration_helper(dict_, [], [])


@lru_cache(maxsize=1)
def _get_main_module_name():
    """ Get __main__ module name """
    from src.utils import ROOT_PATH  # Prevent circular dependecy

    file_name = sys.argv[0]

    try:
        file_name = str(Path(file_name).relative_to(ROOT_PATH))
    except ValueError:
        # `file_name` not relative to `ROOT_PATH`
        pass

    no_extension = file_name.split('.')[0]
    return no_extension.replace('/', '.')


def _get_module_name(func):
    """ Get the name of a module as expressed by it's absolute path.

    Args:
        func (callable): Callable to be inspected.

    Returns:
        keys (list of str): Full path of the module.
        print_name (str): Short name of the module for logging.
    """
    module_keys = inspect.getmodule(func).__name__.split('.')
    if module_keys == ['__main__']:
        module_keys = _get_main_module_name().split('.')
    module_keys = [k for k in module_keys if k != '']
    keys = module_keys + func.__qualname__.split('.')

    if len(module_keys) > 0:
        print_name = module_keys[-1] + '.' + func.__qualname__
    else:
        print_name = func.__qualname__

    return keys, print_name


def _check_configuration_helper(dict_, keys, trace):
    """ Recursive helper of ``_check_configuration``.

    Args:
        dict_ (dict): Parsed dict to check
        keys (list): Current key route in ``dict_``
    """

    if not isinstance(dict_, dict):
        # Recursive function walked up the chain and never found a @configurable
        trace.reverse()
        raise TypeError('Path %s does not contain @configurable.\n' % (keys,) +
                        'Traceback (most recent call last):\n\t%s' % ('\n\t'.join(trace),))

    if len(keys) >= 2:
        # CASE: Function
        # For example:
        #   keys = ['random', 'seed']
        #   module_path = 'random'
        #   function = random.seed
        try:
            # Try to import a function
            module_path = '.'.join(keys[:-1])
            if module_path == _get_main_module_name():
                module_path = '__main__'
            module = import_module(module_path)
            if hasattr(module, keys[-1]):
                function = getattr(module, keys[-1])
                # TODO: Inspect and check if the required parameters exist
                if (hasattr(function, '_configurable')):
                    absolute_keys = _get_module_name(function)[0]
                    if keys != absolute_keys:
                        raise TypeError(
                            'The module path must be absolute: %s vs %s' % (keys, absolute_keys))
                    return
        except ImportError as e:
            trace += ['Module %s ImportError: ' % (module_path,) + str(e)]

    if len(keys) >= 3:
        # CASE: Class
        # For example:
        #   keys = ['nn', 'BatchNorm1d', '__init__']
        #   module_path = 'nn'
        #   class_ = nn.BatchNorm1d
        #   function = nn.BatchNorm1d.__init__
        try:
            module_path = '.'.join(keys[:-2])
            if module_path == _get_main_module_name():
                module_path = '__main__'
            module = import_module(module_path)
            if hasattr(module, keys[-2]):
                class_ = getattr(module, keys[-2])
                if hasattr(class_, keys[-1]):
                    function = getattr(class_, keys[-1])
                    if (hasattr(function, '_configurable')):
                        # NOTE: ``_get_module_name`` is used by configurable for identification;
                        # therefore, enabling us to close the loop with verification.
                        absolute_keys = _get_module_name(function)[0]
                        if keys != absolute_keys:
                            raise TypeError('The module path must be absolute: %s vs %s' %
                                            (keys, absolute_keys))
                        return
        except ImportError as e:
            trace += ['ImportError (%s): ' % (module_path,) + repr(e)]

    for key in dict_:
        # Recusively check every key in ``dict_``
        _check_configuration_helper(dict_[key], keys[:] + [key], trace[:])


def add_config(dict_):
    """ Add configuration to the global configuration.

    Args:
        dict_ (dict): configuration to add

    Returns: None

    Raises:
        (TypeError): module names are formatted improperly
        (TypeError): duplicate functions/modules/packages are defined

    Example:
        >>> import pprint
        >>> pprint.pprint([[1, 2]])
        [[1, 2]]
        >>>
        >>> # Configure `pprint` to use a `width` of `2`
        >>> pprint.pprint = configurable(pprint.pprint)
        >>> add_config({'pprint.pprint': {'width': 2}})
        >>> pprint.pprint([[1, 2]])
        [[1,
          2]]
    """
    global _configuration
    parsed = _parse_configuration(dict_)
    _check_configuration(parsed)
    _dict_merge(_configuration, parsed, overwrite=True)
    _configuration = _KeyListDictionary(_configuration)


def log_config():
    """ Log the current global configuration. """
    logger.info('Global configuration:\n%s', pretty_printer.pformat(_configuration))


def get_config():
    """ Get the current global configuration.

    NOTE: It'd be an antipattern to use this functionality to set or get the configured parameters.
    MOTIVATION: This functionality is intended for releasing the current configuration for logging.

    Returns:
        (dict): The current dictionary.

    """
    return _configuration


def clear_config():
    """ Clear the global configuration """
    global _configuration
    _configuration = _KeyListDictionary()


def _merge_args(parameters, args, kwargs, default_kwargs, print_name=''):
    """ Merge ``func`` ``args`` and ``kwargs`` with ``other_kwargs``

    The ``_merge_args`` prefers ``kwargs`` and ``args`` over ``other_kwargs``.

    Args:
        parameters (list of inspect.Parameter): module that accepts ``args`` and ``kwargs``
        args (list of any): Arguments accepted by ``func``.
        kwargs (dict of any): Key-word arguments accepted by ``func``.
        default_kwargs (dict of any): Default key-word arguments accepted by ``func`` to merge.
        print_name (str, optional): Module name to print with warnings.

    Returns:
        (dict): kwargs merging ``args``, ``kwargs``, and ``other_kwargs``
    """
    default_kwargs = default_kwargs.copy()

    # Delete ``other_kwargs`` that conflict with ``args``
    # Positional arguments must come before key word arguments
    for i, arg in enumerate(args):
        if i >= len(parameters):
            raise TypeError("Too many arguments (%d > %d) passed." % (len(args), len(parameters)))

        if parameters[i].kind == parameters[i].VAR_POSITIONAL:
            # Rest of the args are absorbed by VAR_POSITIONAL (e.g. ``*args``)
            break

        if (parameters[i].kind == parameters[i].POSITIONAL_ONLY or
                parameters[i].kind == parameters[i].POSITIONAL_OR_KEYWORD):
            if parameters[i].name in default_kwargs:
                value = default_kwargs[parameters[i].name]
                if value != arg:
                    # TODO: Do not repeat this warning and warn that this message will not be
                    # repeated
                    logger.warning('Overwriting configured argument ``%s=%s`` in module %s with %s'
                                   % (parameters[i].name, value, print_name, arg))
                del default_kwargs[parameters[i].name]

    for key, value in kwargs.items():
        if key in default_kwargs and value != default_kwargs[key]:
            # TODO: Do not repeat this warning and warn that this message will not be
            # repeated
            logger.warning('Overwriting configured argument ``%s=%s`` in module %s with %s' %
                           (key, default_kwargs[key], print_name, value))

    default_kwargs.update(kwargs)

    return args, default_kwargs


def configurable(func):
    """ Decorater enables configuring module arguments and storing module argument calls.

    Decorator enables one to set the arguments of a module via a global configuration. The decorator
    also stores the parameters the decorated function was called with.

    Args:
        None

    Returns:
        (callable): Decorated function
    """

    @wraps(func)
    def decorator(*args, **kwargs):
        global _configuration

        # Get the module name
        keys, print_name = _get_module_name(func)

        # Get the module config
        config = _configuration[keys] if keys in _configuration else {}  # Get default
        if len(config) == 0:
            # TODO: Do not repeat this warning and warn that this message will not be
            # repeated
            logger.warning('No config for `%s` (`%s`)', print_name, '.'.join(keys))

        # Print name is used for logger
        if not isinstance(config, dict):
            raise ValueError('%s config must be a dict of arguments', print_name)

        parameters = list(inspect.signature(func).parameters.values())
        args, kwargs = _merge_args(parameters, args, kwargs, config, print_name)

        return func(*args, **kwargs)

    # Add a flag to the func; enabling us to check if a function has the configurable decorator.
    decorator._configurable = True

    return decorator

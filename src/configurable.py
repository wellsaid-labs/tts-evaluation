"""
Manages a global namespaced configuration.
TODO: Look into implementing configurable without decoraters:
  - Trace all function calls and intercept the call
  - Apply a decorator recursively too all modules in os.cwd()
    import pkgutil
    import os
    for loader, module_name, is_pkg in pkgutil.walk_packages([os.getcwd()]):
        print(module_name)
    http://code.activestate.com/recipes/577742-apply-decorators-to-all-functions-in-a-module/
"""
from functools import reduce

import inspect
import logging
import operator
import sys
import pprint
from importlib import import_module

import wrapt

pretty_printer = pprint.PrettyPrinter(indent=4)
logger = logging.getLogger(__name__)


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


# Private configuration for all modules in the Python repository
# DO NOT IMPORT. Use @configurable instead.
_configuration = _KeyListDictionary()


def _dict_merge(dict_, merge_dict, overwrite=False):
    """ Recursive `dict` merge. `dict_merge` recurses down into dicts nested to an arbitrary depth,
    updating keys. The `merge_dict` is merged into `dict_`.
    Args:
      dict_ (dict) dict onto which the merge is executed
      merge_dict (dict) dict merged into dict
    """
    for key in merge_dict:
        if key in dict_ and isinstance(dict_[key], dict):
            _dict_merge(dict_[key], merge_dict[key], overwrite)
        elif overwrite and key in dict_:
            dict_[key] = merge_dict[key]
        elif key not in dict_:
            dict_[key] = merge_dict[key]


def _parse_configuration(dict_):
    """
    Transform some `dict_` into a deep _KeyListDictionary that allows for module look ups.
    NOTE: interprets dict_ keys as python `dotted module names`.
    Example:
        `dict_`:
            {
              'abc.abc': {
                'cda': 'abc
              }
            }
        Returns:
            {
              'abc': {
                'abc': {
                  'cda': 'abc
                }
              }
            }
    """
    parsed = {}
    _parse_configuration_helper(dict_, parsed)
    return parsed


def _parse_configuration_helper(dict_, new_dict):
    """ Recursive helper to _parse_configuration """
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


def _check_configuration(dict_, keys=[]):
    """ Check the parsed configuration every module that it points too exists with @configurable.
    Cases to handle recursively:
        {
            'torchutils.nn': {
                'seq_encoder.SeqEncoder.__init__': {
                    'bidirectional': True,
                },
                'attention.Attention.__init__.attention_type': 'general',
            }
        }
    """
    if not isinstance(dict_, dict):
        # Recursive function walked up the chain and never found a @configurable
        logger.warn("""
Path %s does not contain @configurable.

FALSE POSITIVES:
- For modules in __main__, this check can be ignored. (TODO)
- For modules in external libraries, this check can be ignored. (TODO)
        """.strip(), keys)
        return

    if len(keys) >= 2:
        # Scenario: Function
        try:
            module_path = '.'.join(keys[:-1])
            module = import_module(module_path)
            if hasattr(module, keys[-1]):
                function = getattr(module, keys[-1])
                # `is True` to avoid truthy values
                if (hasattr(function, '__wrapped__')):  # TODO: Find a better check
                    return
        except (ImportError, AttributeError):
            pass

    if len(keys) >= 3:
        # Scenario: Class
        try:
            module_path = '.'.join(keys[:-2])
            # TODO: Replace module_path with ``__main__`` if it matches
            module = import_module(module_path)
            if hasattr(module, keys[-2]):
                class_ = getattr(module, keys[-2])
                function = getattr(class_, keys[-1])
                if (hasattr(function, '__wrapped__')):
                    return
        except (ImportError, AttributeError):
            pass

    for key in dict_:
        _check_configuration(dict_[key], keys[:] + [key])


def add_config(dict_):
    """
    Add configuration to the global configuration.
    Example:
        `dict_`=
              {
                'torchutils': {
                  'models': {
                    'decoder_rnn.DecoderRNN.__init__': {
                      'embedding_size': 32
                      'rnn_size': 32
                      'n_layers': 1
                      'rnn_cell': 'gru'
                      'embedding_dropout': 0.0
                      'intra_layer_dropout': 0.0
                    }
                  }
                }
              }
    Args:
        dict_ (dict): configuration to add
        is_log (bool): Note the configuration log can be verbose. If false, do not log the added
            configuration.
    Returns: None
    Raises:
        (TypeError) module names (keys) are formatted improperly (Example: 'torchutils..models')
        (TypeError) duplicate functions/modules/packages are defined
    """
    global _configuration
    parsed = _parse_configuration(dict_)
    logger.info('Checking configuration...')
    _check_configuration(parsed)
    _dict_merge(_configuration, parsed, overwrite=True)
    _configuration = _KeyListDictionary(_configuration)
    logger.info('Configuration checked.')


def log_config():
    """
    Log the global configuration
    """
    logger.info('Global configuration:')
    logging.info(pretty_printer.pformat(_configuration))


def clear_config():
    """
    Clear the global configuration
    Returns: None
    """
    global _configuration
    _configuration = _KeyListDictionary()


def _get_main_module_name():
    """ Get __main__ module name """
    file_name = sys.argv[0]
    no_extension = file_name.split('.')[0]
    return no_extension.replace('/', '.')


def _get_module_name(func):
    """ Get the name of a module. Handles `__main__` by inspecting sys.argv[0]. """
    module = inspect.getmodule(func)
    if module.__name__ == '__main__':
        return _get_main_module_name()
    else:
        return module.__name__


@wrapt.decorator
def configurable(func, instance, args, kwargs):
    """
    Decorator peeks @ the global configuration that defines arguments for some functions. The
    arguments and key word arguments passed to the function are merged with the globally defined
    arguments.
    Args/Return are defined by `wrapt.decorator`.
    """
    global _configuration
    parameters = inspect.signature(func).parameters
    module_keys = _get_module_name(func).split('.')
    keys = module_keys + func.__qualname__.split('.')
    print_name = module_keys[-1] + '.' + func.__qualname__
    default = _configuration[keys] if keys in _configuration else {}  # Get default
    if not isinstance(default, dict):
        logger.info('%s:%s config malformed must be a dict of arguments', print_name,
                    '.'.join(keys))
    merged = default.copy()
    merged.update(kwargs)  # Add kwargs
    # Add args
    args = list(args)
    for parameter in parameters:
        if len(args) == 0 or parameters[parameter].kind == parameters[parameter].VAR_POSITIONAL:
            break
        merged[parameter] = args.pop(0)
        # No POSITIONAL_ONLY arguments
        # https://docs.python.org/3/library/inspect.html#inspect.Parameter
        assert parameter not in kwargs, "Python is broken. Args overwriting kwargs."

    try:
        if len(default) == 0:
            logger.info('%s no config for: %s', print_name, '.'.join(keys))
        # TODO: Does not print all parameters; FIX
        logger.info('%s was configured with:\n%s', print_name, pretty_printer.pformat(merged))
        return func(*args, **merged)
    except TypeError as error:
        logger.info('%s was passed defaults: %s', print_name, default)
        logger.error(error, exc_info=True)
        raise
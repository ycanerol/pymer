#!/usr/bin/env python3
"""
Utility functions for reading the configuration.
"""

import collections
import json
import os
import re


def readjsonfile(filename, is_default_cfg=False):
    """
    Parse JSON file and remove annotative comment when allowed.

    Parameters
    ----------
    filename : str
        Path to a JSON file to parse
    is_default_cfg : bool, optional
        Treat as user or default configuration file. Default is False

    Returns
    -------
    data : dict
        Parsed configuration

    Raises
    ------
    AttributeError
        If parsing of JSON file failed due to a syntax error

    Notes
    -----
    The default config is allowed to have annotative comments which are
    typically illegal in JSON. This enables some descriptions in the default
    config file.
    """
    if not is_default_cfg:
        if not os.path.isfile(filename) or os.stat(filename).st_size <= 0:
            return {}

    with open(filename, 'r') as cfile:
        data = cfile.read()

    # Remove annotative comments in default JSON (don't hate)
    if is_default_cfg:
        data = re.sub(r"//.*$", "", data, flags=re.M)

    try:
        data = json.loads(data)
    except json.JSONDecodeError as je:
        apath = os.path.realpath(filename)
        raise AttributeError('Invalid syntax in the configuration file '
                             f'\'{apath}\':\n{str(je)}') from None
    return data


def nestedupdate(d, u):
    """
    Nested update for dictionaries.

    Parameters
    ----------
    d : dict
        Source dictionary to update
    u : dict
        Update dictionary with altered values
    """
    for k, v in u.items():
        if isinstance(v, collections.Mapping):
            d[k] = nestedupdate(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def cache_config():
    """
    Decorator for caching the loaded config files.
    """
    def _init(obj):
        defaults = readjsonfile('defaultconfig.json', is_default_cfg=True)
        user = readjsonfile(os.path.expanduser('~/.pymer'))
        setattr(obj, 'cfg', nestedupdate(defaults, user))
        return obj
    return _init

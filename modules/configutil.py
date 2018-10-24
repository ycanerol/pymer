#!/usr/bin/env python3
"""
Utility functions for reading the configuration.
"""
import collections
import json
import os
import re
from pathlib import Path
import sys


def readjsonfile(filename, required=False, allow_comments=False):
    """
    Parse JSON file and strip off of annotative comments if allowed.

    Parameters
    ----------
    filename : str
        Path to a JSON file to parse
    required : bool, optional
        If False, return empty dict if file does not exist or is empty
    allow_comments : bool, optional
        Allow annotative comments. Default is False

    Returns
    -------
    data : dict
        Parsed JSON file

    Raises
    ------
    AttributeError
        If parsing of JSON file failed due to a syntax error
    """
    if not required:
        if not os.path.isfile(filename) or os.stat(filename).st_size <= 0:
            return {}

    with open(filename, 'r') as cfile:
        data = cfile.read()

    # Remove annotative comments in default JSON (don't hate)
    if allow_comments:
        data = re.sub(r"//.*$", "", data, flags=re.M)

    try:
        data = json.loads(data)
    except json.JSONDecodeError as je:
        apath = os.path.realpath(filename)
        raise AttributeError('Invalid syntax in the JSON file '
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
        root_path = Path(sys.modules[__name__].__file__).parents[1]
        dflt_path = root_path.joinpath('pymer_config_default.json')
        defaults = readjsonfile(dflt_path, is_default_cfg=True)
        user = readjsonfile(os.path.expanduser('~/.pymer_config'))
        setattr(obj, 'cfg', nestedupdate(defaults, user))
        return obj
    return _init

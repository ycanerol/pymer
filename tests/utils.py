#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
General utilities for testing
"""
import sys
from pathlib import Path


def get_test_path():
    """
    Retrieve the absolute file path to the tests directory.
    """
    return Path(sys.modules[__name__].__file__).parents[0]

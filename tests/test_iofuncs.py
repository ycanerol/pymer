#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test code for pymer.

Usage
=====

From the root directory of pymer run on the command line
``py.test -v tests``

"""
from pymer.modules import iofuncs as iof

import utils


def test_getstimname():
    """
    Test retrieval of stimulus name.
    """
    exp_path = utils.get_test_path().joinpath('test_exp_dir')

    assert iof.getstimname(exp_path, 6) == '6_checkerflicker5x5bw1blink'
    assert iof.getstimname(exp_path, 7) == '7_checkerflicker5x5bw2blinks'
    assert iof.getstimname(exp_path, 10) == '10_checkerflicker_2x2bw_4blinks'
    assert iof.getstimname(exp_path, 1) == '1_spontaneous_dark_lowl1'

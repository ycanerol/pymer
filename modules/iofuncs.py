#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 16:55:33 2018

@author: ycan

Functions related to reading/writing files.
"""

import os
import glob


def exp_dir_fixer(exp_name):
    """
    Convert short experiment name into full path. If input is already
    a full path it will be returned as is.

    Following are some valid inputs:
        Erol_20171122_fe_re_fp
        20171122_fe_re_fp
        20171122_f
        20171122
    They all will be turned into the same full path:
        /home/ycan/Documents/data/Erol_20171122_252MEA_fr_re_fp

    """
    exp_dir = str(exp_name)
    if not os.path.isdir(exp_dir):
        exp_dir = os.path.join('/home/ycan/Documents/data/',
                               exp_name)
        if not os.path.isdir(exp_dir):
            exp_dir = '/home/ycan/Documents/data/Erol_'+exp_name
            if not os.path.isdir(exp_dir):
                files = glob.glob(exp_dir+'*')

                if len(files) > 1:
                    raise ValueError('Multiple files'
                                     'found matching'
                                     ' pattern: {}\n {}'.format(exp_name,
                                                                files))
                elif len(files) == 0:
                    raise ValueError('No files matching'
                                     ' pattern: {}'.format(exp_dir))
                else:
                    exp_dir = files[0]
    return exp_dir

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 16:55:33 2018

@author: ycan

Functions related to reading/writing files.
"""

import os
import glob
import numpy as np


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
            if exp_name[0] == '2':
                exp_dir = '/home/ycan/Documents/data/Erol_'+exp_name
            if not os.path.isdir(exp_dir):
                files = glob.glob(exp_dir+'*')

                if len(files) > 1:
                    raise ValueError('Multiple folders'
                                     'found matching'
                                     ' pattern: {}\n {}'.format(exp_name,
                                                                files))
                elif len(files) == 0:
                    raise ValueError('No folders matching'
                                     ' pattern: {}'.format(exp_dir))
                else:
                    exp_dir = files[0]
    return exp_dir


def loadh5(path):
    """
    Load data from h5 file into workspace.

    Usage:
        locals().update(miscfuncs.loadh5(path))

    h5py module returns an HDF file object in memory when data is
    read. This is not convenient for data manipulation, variables
    should be in the active namespace. For this, loadh5 function
    reads the HDF file and returns a dictionary containing all
    the variable and variable name pairs. This dictionary then
    can be used for loading the variables into workspace.

    Parameters
        path:
            Full path to the .h5 file to be read.
    Returns
        data_in_dict:
            All of the saved variables in a dictionary. This
            should be used with locals().update(data_in_dict)
            in order to load all of the data into main namespace.
    """
    import h5py
    data_in_dict = {}
    f = h5py.File(path, mode='r')
    # Get all the variable names that were saved
    keys = list(f.keys())
    for key in keys:
        item = f[key]
        # To load numpy arrays [:] trick is needed
        try:
            item = item[:]
        except ValueError:
            # This is required to load scalar values into workspace,
            # otherwise they appear as HDF5 dataset objects which are
            # not visible in variable explorer.
            try:
                item = np.asscalar(np.array(item))
            except AttributeError:
                print('%s may not be loaded properly into namespace' % key)
        data_in_dict[key] = item
    # Some variables (e.g. STAs) are stored as lists originally
    # but saving to and loading from HDF file converts them to
    # numpy arrays with one additional dimension. To revert
    # this, we need to turn them back into lists with list()
    # function. The variables that should be converted are
    # to be kept in list_items.
    list_items = ['stas', 'max_inds']

    for list_item in list_items:
        if list_item in keys:
            data_in_dict[list_item] = list(data_in_dict[list_item])
    f.close()
    return data_in_dict


def stimname(exp_name, stim_nr):
    """
    Returns the stimulus name for a given experiment and stimulus
    number from parameters.txt file.
    """
    exp_dir = exp_dir_fixer(exp_name)
    with open(os.path.join(exp_dir, 'parameters.txt')) as f:
        lines = f.readlines()
    lines = lines[1:]
    name = None
    for line in lines:
        if line.startswith('%s_' % stim_nr):
            name = line[:-4].strip(' ')
    return name

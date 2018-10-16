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

import configutil as cutil


# Some variables (e.g. STAs) are stored as lists originally
# but saving to and loading from HDF/npz file converts them to
# numpy arrays with one additional dimension. To revert
# this, we need to turn them back into lists with list()
# function. Any variables that is normally a list should be
# kept in list_of_lists.

list_of_lists = ['stas', 'max_inds', 'all_frs', 'all_parameters', 'fits']


@cutil.cache_config()
def config(key, default=None):
    """
    Retrieve value from loaded config.

    Parameters
    ----------
    key : int or str
        Key in the config dictionary
    default : int or str, optional
        Default value to return if key is not in config. Default is None

    Returns
    -------
    Value of the config for the given key.

    Notes
    -----
    See 'defaultconfig.json' for more information.
    """
    return config.cfg.get(key, default)


def exp_dir_fixer(exp_name):
    """
    Convert short experiment name into full path. If input is already
    a full path it will be returned as is.

    Following are some valid inputs:
        <valid_prefix>_20171122_fe_re_fp
        20171122_fe_re_fp
        20171122_f
        20171122
    They all will be turned into the same full path:
        <root_experiment_dir>/<prefix>_20171122_252MEA_fr_re_fp

    """
    if config('root_experiment_dir') is None:
        raise ValueError('Invalid root experiment directory')

    exp_dir = str(exp_name)
    for s in [''] + config('experiment_prefixes'):
        exp_name = s + exp_name
        if not os.path.isdir(exp_dir):
            exp_dir = os.path.join(config('root_experiment_dir'),
                                   exp_name)
            if not os.path.isdir(exp_dir):

                files = glob.glob(exp_dir+'*')

                if len(files) > 1:
                    raise ValueError('Multiple folders'
                                     'found matching'
                                     ' pattern: {}\n {}'.format(exp_name,
                                                                files))
                elif len(files) == 0:
                    if exp_name[0] == '2':
                        continue
                    raise ValueError('No folders matching'
                                     ' pattern: {}'.format(exp_dir))
                else:
                    exp_dir = files[0]
    return exp_dir


def loadh5(path):
    """
    Load data from h5 file in a dictionary.

    Usage:
        data = loadh5(('20171116', 6))
        stas = data['stas']

    Usage:
        locals().update(loadh5(path))

    h5py module returns an HDF file object in memory when data is
    read. This is not convenient for data manipulation, variables
    should be in the active namespace. For this, loadh5 function
    reads the HDF file and returns a dictionary containing all
    the variable and variable name pairs. This dictionary then
    can be used for loading the variables into workspace.

    Parameters
        path:
            Full path to the .h5 file to be read.

            It can also be a tuple containing experiment name and stimulus
            number, in which case _data.h5 file will be used.
    Returns
        data_in_dict:
            All of the saved variables in a dictionary. This
            can be used with locals().update(data_in_dict)
            in order to load all of the data into main namespace.
    """
    import h5py

    if isinstance(path, tuple):
        exp_name = path[0]
        stimnr = str(path[1])
        exp_dir = exp_dir_fixer(exp_name)
        stim_name = getstimname(exp_dir, stimnr)
        path = os.path.join(exp_dir, 'data_analysis', stim_name,
                            stimnr+'_data.h5')

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

    list_items = list_of_lists

    for list_item in list_items:
        if list_item in keys:
            data_in_dict[list_item] = list(data_in_dict[list_item])
    f.close()
    return data_in_dict


def load(exp_name, stimnr, fname=None):
    """
    Load data from .npz file to a dictionary, while fixing
    lists, scalars and strings.

    If the filename is something other than <stimnr>_data.npz,
    it can be supplied via fname parameter.

    locals().update(data) will load all of the variables in to
    current workspace. This does not work within functions.

    """
    stimnr = str(stimnr)
    exp_dir = exp_dir_fixer(exp_name)
    stim_name = getstimname(exp_dir, stimnr)
    if not fname:
        fname = stimnr+'_data.npz'
    path = os.path.join(exp_dir, 'data_analysis', stim_name,
                        fname)

    data_in_dict = {}
    with np.load(path) as f:
        # Get all the variable names that were saved
        keys = list(f.keys())
        for key in keys:
            item = f[key]
            if item.shape == ():
                item = np.asscalar(item)
            data_in_dict[key] = item
    # Some variables (e.g. STAs) are stored as lists originally
    # but saving to and loading from npz file converts them to
    # numpy arrays with one additional dimension. To revert
    # this, we need to turn them back into lists with list()
    # function. The variables that should be converted are
    # to be kept in list_items.
    list_items = list_of_lists

    for list_item in list_items:
        if list_item in keys:
            data_in_dict[list_item] = list(data_in_dict[list_item])
    return data_in_dict


def getstimname(exp_name, stim_nr):
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
            name = line[:-4].strip()
        # In case the stimulus name is in the format 01_spontaneous
        elif line.startswith('0%s' % stim_nr):
            name = line[1:-4].strip()
    if name:
        return name
    else:
        raise IndexError('Stimulus {} does not exist in experiment:'
                         '\n{}'.format(stim_nr, exp_dir))


def readmat(matfile):
    """
    Read a given .mat file and return the contents in a dictionary.

    .mat files that are v>7.3 and v<7.3 must be treated differently.
    """
    import scipy.io
    data = {}
    try:
        f = scipy.io.matlab.loadmat(matfile)
        useh5 = False
    except NotImplementedError:
        useh5 = True

    if not useh5:
        for key, item in f.items():
            if not str(key).startswith('__'):
                if str(item.dtype).startswith('<U'):
                    item = str(item)
                else:
                    item = np.squeeze(item)
                data.update({key: item})
        return data
    else:
        import h5py
        with h5py.File(matfile, mode='r') as f:
            for key in f.keys():
                data.update({key: np.squeeze(f[key])})
        return data

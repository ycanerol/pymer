#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 14:58:16 2017

@author: ycan

Collection of functions that are used for various stages of analysis
"""
import numpy as np
import matplotlib.pyplot as plt
import glob


def svd(sta):
    # Perform singular value decomposition on STA
    # As described by Gauthier et al. 2009

    # Put temporal component of each pixel in one row
    M = sta.reshape((sta.shape[0]*sta.shape[1], sta.shape[2]))
    # Get U (spatial) and V (temporal) from SVD which returns UDV*
    u, _, v = np.linalg.svd(M)
    # First column of U contains the primary spatial component
    # Second component also gives interesting results sometimes so we look at
    # that as well.
    sp1 = u[:, 0].reshape((sta.shape[0], sta.shape[1]))
    sp2 = u[:, 1].reshape((sta.shape[0], sta.shape[1]))
    # V contains the matching temporal components in its rows (b/c it's
    # transposed).
    t1 = v[0, :]
    t2 = v[1, :]

    # Flip so that the spatial component is positive
    if np.max(np.abs(sp1)) != np.max(sp1):
        sp1 = -sp1
        t1 = -t1
    if np.max(np.abs(sp2)) != np.max(sp2):
        sp2 = -sp2
        t2 = -t2

    return sp1, sp2, t1, t2, u, v


def show_sta(sta, max_i, f_size=10):
    plt.plot(figsize=(12, 12), dpi=200)
    sta_min = np.min(sta)
    sta_max = np.max(sta)

    for i in range(20):
        plt.subplot(4, 5, i+1)

        plt.imshow(sta[:, :, i], vmin=sta_min, vmax=sta_max, cmap='Greys')
        plt.axis('off')
    plt.show()


def readexps(directory, test=False):
    # TODO: stimulus_order needs to be parametrized/automated properly
    stimulus_order = 5
    file_paths = glob.glob(directory +
                           '*/analyzed/{}*.npz'.format(stimulus_order))
    file_paths = sorted(file_paths)
    exp_names = [i.split('/')[-3] for i in file_paths]
    clusters = [i.split('/')[-1].split('C')[-1].split('.')[0] for i in file_paths]

    files = np.array([file_paths, exp_names, clusters])

    # Use only one file for testing
    if test:
        files = files[:, np.random.randint(files.shape[1])]
        files = files[:, np.newaxis]

    return files


def ringmask(data, center_px, r):
    # Create a ring shaped mask  in spatial dimention
    # and apply to the STA along time axis

    # HINT: "True" masks the value out
    mask = np.array([True]*data.size).reshape(data.shape)

    cx, cy, _ = center_px

    # Check if the specified ring size is larger than the shape
    outofbounds = (cx+r > data.shape[0]-1 or cx-r < 0 or
                   cx+r > data.shape[0]-1 or cx-r < 0)

    mask[cx-r:cx+r+1, cy-r:cy+r+1, :] = False
    mask[cx-(r-1):cx+(r), cy-(r-1):cy+(r), :] = True

    masked_data = np.ma.array(data, mask=mask)

    if outofbounds:
        masked_data = None

    return masked_data, outofbounds


def cut_around_center(sta_original, max_i_o, f_size):
    if f_size+2 > (sta_original.shape[0])/2 or f_size+2 > (sta_original.shape[1])/2:
        raise ValueError('Frame size is larger than STA dimensions')
    if f_size is not 0:
        sta = sta_original[max_i_o[0]-f_size:max_i_o[0]+f_size+1,
                           max_i_o[1]-f_size:max_i_o[1]+f_size+1,
                           :]
        max_i = np.append([f_size]*2, max_i_o[2])

    else:
        sta = sta_original
        max_i = max_i_o
    return sta, max_i


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
    list_items = ['stas', 'max_inds', 'clusterids']

    for list_item in list_items:
        if list_item in keys:
            data_in_dict[list_item] = list(data_in_dict[list_item])
    f.close()
    return data_in_dict

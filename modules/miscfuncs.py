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
    try:
        from lnp_checkerflicker import check_max_i
    except ImportError:
        import sys
        sys.path.append('/home/ycan/Documents/scripts/modules')
        from lnp_checkerflicker import check_max_i
    if f_size is not 0:
        max_i_o = check_max_i(sta_original, max_i_o, f_size=f_size)
        sta = sta_original[max_i_o[0]-f_size:max_i_o[0]+f_size+1,
                           max_i_o[1]-f_size:max_i_o[1]+f_size+1,
                           :]
        max_i = np.append([f_size]*2, max_i_o[2])

    else:
        sta = sta_original
        max_i = max_i_o
    return sta, max_i

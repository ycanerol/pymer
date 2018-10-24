#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Collection of functions that are used for various stages of analysis.
"""
import glob
import numpy as np
import warnings


def binspikes(spiketimes, frametimings):
    """
    Bins spikes according to frametimings, discards spikes before and after
    stimulus presentation

    Parameters:
    ----------
        spiketimes:
            Array containing times of when spikes happen, as output of
            io.read_rasters()
        frametimings:
            Array containing frametimings as output of frametimes.read()
    Returns:
    -------
        spikes:
            Binned array of spikes according to frametimings.

    Notes:
    -----
        Binning according to frametimings results in high count in first and
        last bins since recording is started earlier than stimulus presentation
        and stopped after presentation ends. Therefore the spikes that happen
        before or after stimulus presentation are discarded in the output.
    """
    spikes = np.bincount(np.digitize(spiketimes, frametimings))
    if spikes.shape == (0,):
        # If there are no spikes for a particular cell,
        # set it to an array of zeros.
        spikes = np.zeros((frametimings.shape[0]+1,))
    spikes[0] = 0
    spikes = spikes[:-1]

    # HINT: This might cause problems! Not sure this is the correct way to
    # fix this problem, might cause misalignment of frametimes and spikes
    #
    # If there hasn't been any spikes at the end of the recording, the length
    # of the spikes array will be shorter than the frametimings and this
    # results in not having data at the end of the recording.
    while spikes.shape[0] < frametimings.shape[0]:
        spikes = np.append(spikes, 0)

    return spikes


def staquality(sta):
    """
    Calculates the z-score of the pixel that is furthest away
    from the zero as a measure of STA quality.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        z = (np.max(np.abs(sta)) - sta.mean()) / sta.std()
    return z.astype('float16')


def svd(sta, flip=False):
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

    if flip:
        # Flip so that the spatial component is positive
        if np.max(np.abs(sp1)) != np.max(sp1):
            sp1 = -sp1
            t1 = -t1
        if np.max(np.abs(sp2)) != np.max(sp2):
            sp2 = -sp2
            t2 = -t2

    return sp1, sp2, t1, t2, u, v


def readexps(directory, test=False):
    # TODO: stimulus_order needs to be parametrized/automated properly
    stimulus_order = 5
    file_paths = glob.glob(directory +
                           '*/analyzed/{}*.npz'.format(stimulus_order))
    file_paths = sorted(file_paths)
    exp_names = [i.split('/')[-3] for i in file_paths]
    clusters = [i.split('/')[-1].split('C')[-1].split('.')[0]
                for i in file_paths]

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
    if (f_size+2 > (sta_original.shape[0])/2
            or f_size+2 > (sta_original.shape[1])/2):
        raise ValueError('Frame size is larger than STA dimensions')
    if (max_i_o[0] + f_size > sta_original.shape[0] or
            max_i_o[0] - f_size < 1):
        raise ValueError('Frame is out of the STA dimension')
    if (max_i_o[1] + f_size > sta_original.shape[1] or
            max_i_o[1] - f_size < 1):
        raise ValueError('Frame is out of the STA dimension')

    if len(sta_original.shape) == 3:
        if f_size is not 0:
            sta = sta_original[max_i_o[0]-f_size:max_i_o[0]+f_size+1,
                               max_i_o[1]-f_size:max_i_o[1]+f_size+1,
                               :]
            max_i = np.append([f_size]*2, max_i_o[2])

        else:
            sta = sta_original
            max_i = max_i_o
    elif len(sta_original.shape) == 2:
        sta = sta_original[max_i_o[0]-f_size:max_i_o[0]+f_size+1,
                           max_i_o[1]-f_size:max_i_o[1]+f_size+1]
        max_i = np.append([f_size]*2, max_i_o[2])

    return sta, max_i


def timediff(starttime):
    """
    Calculates elapsed time since a given timepoint, strips the miliseconds
    and returns the result.

    Parameters:
        starttime (datetime.datetime object)
        Timepoint from which difference will be calculated.
    Returns:
        elapsed (datetime.timedelta object)
        Time difference without miliseconds.
    """
    import datetime
    elapsed = datetime.datetime.now()-starttime
    elapsed -= datetime.timedelta(microseconds=elapsed.microseconds)
    return elapsed


def cutstripe(sta, max_i, fsize):
    if max_i[0] - fsize <= 0 or max_i[0] + fsize > sta.shape[0]:
        raise ValueError('Cutting outside the STA range.')
    sta_r = sta[max_i[0]-fsize:max_i[0]+fsize+1, :]
    max_i_r = np.append(fsize, max_i[-1])
    return sta_r, max_i_r

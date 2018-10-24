#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 15:54:03 2017

@author: ycan

Collection of analysis functions
"""
import numpy as np
import os
import warnings

from .. import io as iof


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


def stimulisorter(exp_name):
    """
    Read parameters.txt file and return the stimuli type and
    stimuli numbers in a dictionary.
    """
    possible_stim_names = ['spontaneous', 'onoffsteps', 'fff', 'stripeflicker',
                           'checkerflicker', 'directiongratingsequence',
                           'rotatingstripes', 'frozennoise',
                           'checkerflickerplusmovie', 'OMSpatches', 'OMB',
                           'saccadegrating']
    sorted_stimuli = {key: [] for key in possible_stim_names}
    exp_dir = iof.exp_dir_fixer(exp_name)

    file = open(os.path.join(exp_dir, 'parameters.txt'), 'r')

    for line in file:
        for stimname in possible_stim_names:
            if line.find(stimname) > 0:
                stimnr = int(line.split('_')[0])
                toadd = sorted_stimuli[stimname]
                toadd = toadd.append(stimnr)
    return sorted_stimuli

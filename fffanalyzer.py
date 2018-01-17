#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 15:21:50 2018

@author: ycan
"""
import numpy as np
import os
import analysis_scripts as asc
import iofuncs as iof
from randpy import randpy
import plotfuncs as plf
import matplotlib.pyplot as plt


def fffanalyzer(exp_name, stimnr):
    """
    Analyzes and plots data from full field flicker
    stimulus.
    """
    stimnr = str(stimnr)
    exp_dir = iof.exp_dir_fixer(exp_name)
    exp_name = os.path.split(exp_dir)[-1]

    stimname = iof.getstimname(exp_name, stimnr)

    clusters, metadata = asc.read_ods(exp_dir)

    parameters = asc.read_parameters(exp_dir, stimnr)

    clusterids = plf.clusters_to_ids(clusters)

    if parameters['stixelheight'] < 600 or parameters['stixelwidth'] < 800:
        raise ValueError('Make sure the stimulus is full field flicker.')

    nblinks = parameters['Nblinks']
    try:
        bw = parameters['blackwhite']
    except KeyError:
        bw = False

    try:
        seed = parameters['seed']
    except KeyError:
        seed = -10000

    if nblinks == 1:
        ft_on, ft_off = asc.readframetimes(exp_dir, stimnr,
                                           returnoffsets=True)
        # Initialize empty array twice the size of one of them, assign
        # value from on or off to every other element.
        frametimings = np.empty(ft_on.shape[0]*2, dtype=float)
        frametimings[::2] = ft_on
        frametimings[1::2] = ft_off
        # Set filter length so that temporal filter is ~600 ms. The unit
        # here is number of frames.
        filter_length = 40
    elif nblinks == 2:
        frametimings = asc.readframetimes(exp_dir, stimnr)
        filter_length = 20
    else:
        raise ValueError('Unexpected value for nblinks.')

    frame_duration = np.average(np.ediff1d(frametimings))
    total_frames = frametimings.shape[0]

    all_spiketimes = []
    # Store spike triggered averages in a list containing correct shaped
    # arrays
    stas = []

    for i in range(len(clusters[:, 0])):
        spiketimes = asc.read_raster(exp_dir, stimnr,
                                     clusters[i, 0], clusters[i, 1])
        spikes = asc.binspikes(spiketimes, frametimings)
        all_spiketimes.append(spikes)
        stas.append(np.zeros(filter_length))

    if bw:
        randnrs, seed = randpy.ran1(seed, total_frames)
        randnrs = [1 if i > .5 else -1 for i in randnrs]
    else:
        randnrs, seed = randpy.gasdev(seed, total_frames)

    stimulus = np.array(randnrs)

    for k in range(filter_length, total_frames-filter_length+1):
        stim_small = stimulus[k-filter_length+1:k+1][::-1]
        for j in range(clusters.shape[0]):
            spikes = all_spiketimes[j]
            if spikes[k] != 0:
                stas[j] += spikes[k]*stim_small
    spikenrs = np.array([spikearr.sum() for spikearr in all_spiketimes])

    plotpath = os.path.join(exp_dir, 'data_analysis',
                            stimname, 'filters')
    if not os.path.isdir(plotpath):
        os.makedirs(plotpath, exist_ok=True)

    t = np.arange(filter_length)*frame_duration*1000

    for i in range(clusters.shape[0]):
        stas[i] = stas[i]/spikenrs[i]
        ax = plt.subplot(111)
        ax.plot(t, stas[i])
        plf.spineless(ax)
        plt.xlabel('Time[ms]')
        plt.title('{}\n{}\n{} Rating: {}{} '
                  'spikes'.format(exp_name, stimname, clusterids[i],
                                  clusters[i, 2], int(spikenrs[i])))
        plt.savefig(os.path.join(plotpath, clusterids[i])+'.svg',
                    format='svg', dpi=300)
        plt.close()

    savepath = os.path.join(os.path.split(plotpath)[0], stimnr+'_data')

    keystosave = ['stas', 'clusters', 'frame_duration', 'all_spiketimes',
                  'stimname', 'total_frames', 'spikenrs', 'bw', 'nblinks',
                  'filter_length', 'exp_name']
    data_in_dict = {}
    for key in keystosave:
        data_in_dict[key] = locals()[key]

    np.savez(savepath, **data_in_dict)

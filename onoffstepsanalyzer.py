#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 18:18:03 2017

@author: ycan
"""
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
import analysis_scripts as asc
import matplotlib as mpl
import plotfuncs as plf

# Durations as they appear in onoff steps parameter file
stim_duration = 60
preframe_duration = 120
# Convert from number of frames to seconds
stim_duration = stim_duration/60
preframe_duration = preframe_duration/60
total_cycle = (stim_duration+preframe_duration)*2
contrast = 1

experiment_dir = '/home/ycan/Documents/data/Erol_20171116_252MEA_sr_le_sp'

onoffstimulus_order = 3

wdir = os.getcwd()
try:
    os.chdir(experiment_dir)
    stimulusname = glob.glob('%s_*.mcd' % onoffstimulus_order)[0]
finally:
    os.chdir(wdir)

stim_names = stimulusname.split('.mcd')[0]
stim_nrs = stimulusname.split('_')[0]

clusters, metadata = asc.read_ods(experiment_dir, cutoff=4)

frametimings = asc.getframetimes(experiment_dir, 3)
# %%
all_spikes = []
for i in range(len(clusters[:, 0])):
    spikes = asc.read_raster(experiment_dir, onoffstimulus_order,
                             clusters[i, 0], clusters[i, 1])
    all_spikes.append(spikes)
    # Find which trial each spike belongs to, and subtract one to be able
    # to use as indices
    trial_indices = np.digitize(spikes, frametimings[::4])-1
    # Discard the spikes that happen before the frames start coming
    spikes = spikes[trial_indices >= 0]
    trial_indices = trial_indices[trial_indices >= 0]
    rasterplot = []
    # Iterate over all the trials, add one to the end for the spikes
    # that might happen after the last trial.
    for j in range(int(np.ceil(frametimings.max()/total_cycle))+1):
        rasterplot.append([])
    # plt.eventplot requires a list containing spikes in each trial separately
    for k in range(len(spikes)):
        trial = trial_indices[k]
        rasterplot[trial].append(spikes[k]-frametimings[::4][trial])

    # Workaround for matplotlib issue #6412.
    # https://github.com/matplotlib/matplotlib/issues/6412
    # If a cell has no spikes for the first trial i.e. the first element of
    # the list is empty, an error is raies due to a plt.eventplot bug.
    if len(rasterplot[0]) == 0:
        rasterplot[0] = [-1]

    fig = plt.figure()
    ax1 = plt.subplot(211)
    plt.eventplot(rasterplot, linewidth=.5, color='r')
    # Set the axis so they align with the rectangles
    plt.axis([0, total_cycle, -1, len(rasterplot)+1])

    # Draw rectangles to represent different parts of the on off steps stimulus
    rect1 = mpl.patches.Rectangle((0, 0),
                                  width=preframe_duration/total_cycle, height=1,
                                  transform=ax1.transAxes, color='k',
                                  alpha=.5)
    rect2 = mpl.patches.Rectangle(((preframe_duration+stim_duration)/total_cycle, 0),
                                  width=preframe_duration/total_cycle, height=1,
                                  transform=ax1.transAxes, color='k',
                                  alpha=.5)
    rect3 = mpl.patches.Rectangle(((2*preframe_duration+stim_duration)/total_cycle, 0),
                                  width=stim_duration/total_cycle, height=1,
                                  transform=ax1.transAxes, color='k',
                                  alpha=1)
    ax1.add_patch(rect1)
    ax1.add_patch(rect2)
    ax1.add_patch(rect3)

    plt.suptitle('{}\n{}'.format(experiment_dir.split('/')[-1],
                                 stim_names))
    plt.title('{}{:0>2} Rating: {}'.format(clusters[i][0],
                                           clusters[i][1],
                                           clusters[i][2]))
    plt.ylabel('Trial')
    ax1.set_xticks([])
    plf.spineless(ax1)

    ax2 = plt.subplot(212)

    # Collect all trials in one array to calculate firing rates
    ras = np.array([])
    for i in range(len(rasterplot)):
        ras = np.append(ras, rasterplot[i])

    bins = 150
    t = np.linspace(0, total_cycle, num=bins)

    # Sort into time bins and count how many spikes happened in each
    fr = np.digitize(ras, t)
    fr1 = np.bincount(fr)
    # Normalize so that units are spikes/s
    fr1 = fr1 * (bins/total_cycle) / len(rasterplot)-1
    # Equalize the length of the two arrays for plotting.
    # np.bincount(x) normally produces x.max()+1 bins
    if fr1.shape[0] == bins+1:
        fr1 = fr1[:-1]
    # If there aren't any spikes at the last trial, the firing rates array is
    # too short and plt.plot raises error.
    while fr1.shape[0] < bins:
        fr1 = np.append(fr1, 0)

    plt.plot(t, fr1)
    plf.spineless(ax2)
    plt.axis([0, total_cycle, fr1.min(), fr1.max()])
    plt.xlabel('Time[s]')
    plt.ylabel('Firing rate[spikes/s]')
    plt.show()
    plt.close()

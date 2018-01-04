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


def onoffanalyzer(experiment_dir, stim_order, stim_duration,
                  preframe_duration, contrast=1):
    """
    Analyze onoffsteps data, plot and save it. Will make a directory
    /data_analysis/<stimulus_name> and save svg [and pdf in subfolder.].

    Parameters:
        experiment_dir:
            Experiment directory. Can be foldername directly under
            /home/ycan/Documents/data/. If not, needs to be a full path.
        stim_order:
            Order of the onoff steps stimulus.
        stim_duration:
            The duration of on or off step presentation, number of frames
        preframes_duration:
            The duration of preframes between on-off steps, in number of frames
        contrast:
            The contrast parameter that is given to stimulator program, default
            is 1.

            Example: for 3_onoffsteps60_preframes120.mcd,
            stim_order = 3
            stim_duration = 60
            preframe_duration = 120.
    Returns:
        Nothing. Will return PSTH in the future.

    """
    # Convert from number of frames to seconds
    stim_duration = stim_duration/60
    preframe_duration = preframe_duration/60
    total_cycle = (stim_duration+preframe_duration)*2
    if not os.path.isdir(experiment_dir):
        experiment_dir = os.path.join('/home/ycan/Documents/data/',
                                      experiment_dir)
    # Normalize path in case there is a trailing slash in the path name
    experiment_dir = os.path.normpath(experiment_dir)
    wdir = os.getcwd()
    try:
        os.chdir(experiment_dir)
        stimulusname = np.sort(glob.glob('%s_*.mcd' % stim_order))[0]
    finally:
        os.chdir(wdir)

    stim_names = stimulusname.split('.mcd')[0]

    clusters, metadata = asc.read_ods(experiment_dir, cutoff=4)

    # The first trial will be discarded by dropping the first four frames
    # If we don't save the original and re-initialize for each cell,
    # frametimings will get smaller over time.
    frametimings_original = asc.readframetimes(experiment_dir,
                                               stim_order)

    for i in range(len(clusters[:, 0])):
        spikes = asc.read_raster(experiment_dir, stim_order,
                                 clusters[i, 0], clusters[i, 1])
        frametimings = frametimings_original
        # Discard all the spikes that happen after the last frame
        spikes = spikes[spikes < frametimings[-1]]
        # Discard the first trial
        spikes = spikes[spikes > frametimings[4]]
        frametimings = frametimings[4:]
        # Find which trial each spike belongs to, and subtract one
        # to be able to use as indices
        trial_indices = np.digitize(spikes, frametimings[::4])-1

        rasterplot = []
        # Iterate over all the trials, create an empty array for each
        for j in range(int(np.ceil(frametimings.max()/total_cycle))):
            rasterplot.append([])
        # plt.eventplot requires a list containing spikes in each
        # trial separately
        for k in range(len(spikes)):
            trial = trial_indices[k]
            rasterplot[trial].append(spikes[k]-frametimings[::4][trial])

        # Workaround for matplotlib issue #6412.
        # https://github.com/matplotlib/matplotlib/issues/6412
        # If a cell has no spikes for the first trial i.e. the first
        # element of the list is empty, an error is raies due to
        # a plt.eventplot bug.
        if len(rasterplot[0]) == 0:
            rasterplot[0] = [-1]
            print('Added -1 to first trial')

        fig = plt.figure()
        ax1 = plt.subplot(211)
        plt.eventplot(rasterplot, linewidth=.5, color='r')
        # Set the axis so they align with the rectangles
        plt.axis([0, total_cycle, -1, len(rasterplot)])

        # Draw rectangles to represent different parts of the on off
        # steps stimulus
        rect1 = mpl.patches.Rectangle((0, 0),
                                      width=preframe_duration/total_cycle,
                                      height=1,
                                      transform=ax1.transAxes, color='k',
                                      alpha=.5)
        rect2 = mpl.patches.Rectangle((preframe_duration/total_cycle, 0),
                                      width=stim_duration/total_cycle,
                                      height=1,
                                      transform=ax1.transAxes, color='k',
                                      alpha=.5*(1-contrast))
        rect3 = mpl.patches.Rectangle(((preframe_duration +
                                        stim_duration)/total_cycle, 0),
                                      width=preframe_duration/total_cycle,
                                      height=1,
                                      transform=ax1.transAxes, color='k',
                                      alpha=.5)
        rect4 = mpl.patches.Rectangle(((2*preframe_duration +
                                        stim_duration)/total_cycle, 0),
                                      width=stim_duration/total_cycle,
                                      height=1,
                                      transform=ax1.transAxes, color='k',
                                      alpha=.5*(1+contrast))
        ax1.add_patch(rect1)
        ax1.add_patch(rect2)
        ax1.add_patch(rect3)
        ax1.add_patch(rect4)

        plt.suptitle('{}\n{}'.format(os.path.split(experiment_dir)[-1],
                                     stim_names))
        plt.title('{:0>3}{:0>2} Rating: {}'.format(clusters[i][0],
                                                   clusters[i][1],
                                                   clusters[i][2]))
        plt.ylabel('Trial')
        ax1.set_xticks([])
        plf.spineless(ax1)

        ax2 = plt.subplot(212)

        # Collect all trials in one array to calculate firing rates
        ras = np.array([])
        for ii in range(len(rasterplot)):
            ras = np.append(ras, rasterplot[ii])

        bins = 150
        t = np.linspace(0, total_cycle, num=bins)

        # Sort into time bins and count how many spikes happened in each
        fr = np.digitize(ras, t)
        fr1 = np.bincount(fr)
        # Normalize so that units are spikes/s
        fr1 = fr1 * (bins/total_cycle) / (len(rasterplot)-1)
        # Equalize the length of the two arrays for plotting.
        # np.bincount(x) normally produces x.max()+1 bins
        if fr1.shape[0] == bins+1:
            fr1 = fr1[:-1]
        # If there aren't any spikes at the last trial, the firing
        # rates array is too short and plt.plot raises error.
        while fr1.shape[0] < bins:
            fr1 = np.append(fr1, 0)

        plt.plot(t, fr1)
        plf.spineless(ax2)
        plt.axis([0, total_cycle, fr1.min(), fr1.max()])
        plt.xlabel('Time[s]')
        plt.ylabel('Firing rate[spikes/s]')

        savedir = os.path.join(experiment_dir, 'data_analysis', stim_names)
        os.makedirs(os.path.join(savedir, 'pdf'), exist_ok=True)

        # Save as svg for looking through data, pdf for
        # inserting into presentations
        plt.savefig(savedir+'/{:0>3}{:0>2}.svg'.format(clusters[i, 0],
                                                       clusters[i, 1]),
                    format='svg')
        plt.savefig(os.path.join(savedir, 'pdf',
                                 '{:0>3}{:0>2}.pdf'.format(clusters[i, 0],
                                                           clusters[i, 1])),
                    format='pdf')
        plt.close()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 18:18:03 2017

@author: ycan
"""
import numpy as np
import os
import matplotlib.pyplot as plt
import analysis_scripts as asc
import matplotlib as mpl
import plotfuncs as plf
import iofuncs as iof


def onoffstepsanalyzer(exp_name, stim_nr):
    """
    Analyze onoffsteps data, plot and save it. Will make a directory
    /data_analysis/<stimulus_name> and save svg [and pdf in subfolder.].

    Parameters:
        exp_name:
            Experiment name.
        stim_nr:
            Order of the onoff steps stimulus.

    """

    exp_dir = iof.exp_dir_fixer(exp_name)

    stim_nr = str(stim_nr)

    exp_name = os.path.split(exp_dir)[-1]

    stimname = iof.getstimname(exp_dir, stim_nr)

    clusters, metadata = asc.read_ods(exp_dir, cutoff=4)

    parameters = asc.read_parameters(exp_dir, stim_nr)

    # Divide by 60 to convert from number of frames to seconds
    stim_duration = parameters['Nframes']/60
    try:
        preframe_duration = parameters['preframes']/60
    except KeyError:
        preframe_duration = 0

    contrast = parameters['contrast']

    total_cycle = (stim_duration+preframe_duration)*2

    # The first trial will be discarded by dropping the first four frames
    # If we don't save the original and re-initialize for each cell,
    # frametimings will get smaller over time.
    frametimings_original = asc.readframetimes(exp_dir, stim_nr)

    savedir = os.path.join(exp_dir, 'data_analysis', stimname)
    os.makedirs(os.path.join(savedir, 'pdf'), exist_ok=True)

    # Collect all firing rates in a list
    all_frs = []

    for i in range(len(clusters[:, 0])):
        spikes = asc.read_raster(exp_dir, stim_nr,
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
        # element of the list is empty, an error is raised due to
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

        plt.suptitle('{}\n{}'.format(exp_name, stimname))
        plt.title('{:0>3}{:0>2} Rating: {}'.format(clusters[i][0],
                                                   clusters[i][1],
                                                   clusters[i][2]))
        plt.ylabel('Trial')
        plt.gca().invert_yaxis()
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

        all_frs.append(fr1)
        plt.plot(t, fr1)
        plf.spineless(ax2)
        plt.axis([0, total_cycle, fr1.min(), fr1.max()])
        plt.xlabel('Time[s]')
        plt.ylabel('Firing rate[spikes/s]')

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

    keystosave = ['clusters', 'total_cycle', 'bins', 'tstep',
                  'stimname', 'stim_duration', 'preframe_duration',
                  'contrast', 'all_frs', 't', 'exp_name']
    data_in_dict = {}
    for key in keystosave:
        data_in_dict[key] = locals()[key]

    np.savez(os.path.join(savedir, stim_nr + '_data'), **data_in_dict)

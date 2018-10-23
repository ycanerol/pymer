#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 18:18:03 2017

@author: ycan
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import analysis_scripts as asc
import plotfuncs as plf
import iofuncs as iof


def spontanalyzer(exp_name, stim_nrs):
    """
    Analyze spontaneous activity, plot and save it. Will make a directory
    /data_analysis/<stimulus_name> and save svg [and pdf in subfolder.].

    """

    exp_dir = iof.exp_dir_fixer(exp_name)

    exp_name = os.path.split(exp_dir)[-1]

    if isinstance(stim_nrs, int):
        stim_nrs = [stim_nrs]
    elif len(stim_nrs) == 0:
        return

    for stim_nr in stim_nrs:
        stim_nr = str(stim_nr)

        stimname = iof.getstimname(exp_dir, stim_nr)

        clusters, _ = asc.read_spikesheet(exp_dir, cutoff=4)

        # Length of chunks we use for dividing the activity for plotting.
        step = 1

        allspikes = []

        for i in range(clusters.shape[0]):
            spikes = asc.read_raster(exp_dir, stim_nr,
                                     clusters[i, 0], clusters[i, 1])
            allspikes.append(spikes)

        # Use the time of the last spike to determine the total recording time.
        last_spike = np.max([np.max(allspikes[i])\
                             for i in range(clusters.shape[0])
                             if len(allspikes[i]) > 0])
        totalrecordingtime = np.int(np.ceil(last_spike) + 1)
        times = np.arange(0, totalrecordingtime, step)

        for i in range(len(clusters[:, 0])):
            spikes = allspikes[i]
            # Find which trial each spike belongs to, and subtract one
            # to be able to use as indices
            trial_indices = np.digitize(spikes, times)-1

            rasterplot = []
            # Iterate over all the trials, create an empty array for each
            for j in range(totalrecordingtime):
                rasterplot.append([])
            # plt.eventplot requires a list containing spikes in each
            # trial separately
            for k in range(len(spikes)):
                trial = trial_indices[k]
                rasterplot[trial].append(spikes[k]-times[trial])

            # Workaround for matplotlib issue #6412.
            # https://github.com/matplotlib/matplotlib/issues/6412
            # If a cell has no spikes for the first trial i.e. the first
            # element of the list is empty, an error is raised due to
            # a plt.eventplot bug.
            if len(rasterplot[0]) == 0:
                rasterplot[0] = [-1]

            plt.figure(figsize=(9, 6))
            ax1 = plt.subplot(111)
            plt.eventplot(rasterplot, linewidth=.5, color='k')
            # Set the axis so they align with the rectangles
            plt.axis([0, step, -1, len(rasterplot)])


            plt.suptitle('{}\n{}'.format(exp_name, stimname))
            plt.title('{:0>3}{:0>2} Rating: {}'.format(clusters[i][0],
                                                       clusters[i][1],
                                                       clusters[i][2]))
            plt.ylabel('Time index')
            plt.xlabel('Time[s]')
            plt.gca().invert_yaxis()
            ax1.set_xticks([0, .5, 1])
            plf.spineless(ax1)

            savedir = os.path.join(exp_dir, 'data_analysis', stimname)
            os.makedirs(os.path.join(savedir, 'pdf'), exist_ok=True)

            # Save as svg for looking through data, pdf for
            # inserting into presentations
            plt.savefig(savedir+'/{:0>3}{:0>2}.svg'.format(clusters[i, 0],
                                                           clusters[i, 1]),
                        format='svg', bbox_inches='tight')
            plt.savefig(os.path.join(savedir, 'pdf',
                                     '{:0>3}'
                                     '{:0>2}.pdf'.format(clusters[i, 0],
                                                         clusters[i, 1])),
                        format='pdf', bbox_inches='tight')
            plt.close()
        print(f'Analysis of {stimname} completed.')

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


def onoffstepsanalyzer(exp_name, stim_nrs):
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

    exp_name = os.path.split(exp_dir)[-1]

    if isinstance(stim_nrs, int):
        stim_nrs = [stim_nrs]

    for stim_nr in stim_nrs:
        stim_nr = str(stim_nr)

        stimname = iof.getstimname(exp_dir, stim_nr)

        clusters, metadata = asc.read_spikesheet(exp_dir, cutoff=4)

        clusterids = plf.clusters_to_ids(clusters)

        parameters = asc.read_parameters(exp_dir, stim_nr)

        refresh_rate = metadata['refresh_rate']

        # Divide by the refresh rate to convert from number of
        # frames to seconds
        stim_duration = parameters['Nframes']/refresh_rate

        preframe_duration = parameters.get('preframes',
                                                   0)/refresh_rate

        contrast = parameters['contrast']

        total_cycle = (stim_duration+preframe_duration)*2

        # Set the bins to be 10 ms
        tstep = 0.01
        bins = int(total_cycle/tstep)+1
        t = np.linspace(0, total_cycle, num=bins)

        # Setup for onoff bias calculation
        onbegin = preframe_duration
        onend = onbegin+stim_duration
        offbegin = onend+preframe_duration
        offend = offbegin+stim_duration

        # Determine the indices for each period
        a = []
        for i in [onbegin, onend, offbegin, offend]:
            yo = np.asscalar(np.where(np.abs(t-i) < tstep/2)[0][-1])
            a.append(yo)

        # To exclude stimulus offset affecting the bias, use
        # last 1 second of preframe period
        prefs = []
        for i in [onbegin-1, onbegin, offbegin-1, offbegin]:
            yo = np.asscalar(np.where(np.abs(t-i) < tstep/2)[0][-1])
            prefs.append(yo)

        onper = slice(a[0], a[1])
        offper = slice(a[2], a[3])

        pref1 = slice(prefs[0], prefs[1])
        pref2 = slice(prefs[2], prefs[3])

        onoffbias = np.empty(clusters.shape[0])
        baselines = np.empty(clusters.shape[0])

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

            plt.figure(figsize=(9, 9))
            ax1 = plt.subplot(211)
            plt.eventplot(rasterplot, linewidth=.5, color='r')
            # Set the axis so they align with the rectangles
            plt.axis([0, total_cycle, -1, len(rasterplot)])

            # Draw rectangles to represent different parts of the on off
            # steps stimulus
            plf.drawonoff(ax1, preframe_duration, stim_duration,
                          contrast=contrast)

            plt.ylabel('Trial')
            plt.gca().invert_yaxis()
            ax1.set_xticks([])
            plf.spineless(ax1)


            # Collect all trials in one array to calculate firing rates
            ras = np.array([])
            for ii in range(len(rasterplot)):
                ras = np.append(ras, rasterplot[ii])

            # Sort into time bins and count how many spikes happened in each
            fr = np.digitize(ras, t)
            fr = np.bincount(fr)
            # Normalize so that units are spikes/s
            fr = fr * (bins/total_cycle) / (len(rasterplot)-1)
            # Equalize the length of the two arrays for plotting.
            # np.bincount(x) normally produces x.max()+1 bins
            if fr.shape[0] == bins+1:
                fr = fr[:-1]
            # If there aren't any spikes at the last trial, the firing
            # rates array is too short and plt.plot raises error.
            while fr.shape[0] < bins:
                fr = np.append(fr, 0)

            prefr = np.append(fr[pref1], fr[pref2])
            baseline = np.median(np.round(prefr))

            fr_corr = fr - baseline

            r_on = np.sum(fr_corr[onper])
            r_off = np.sum(fr_corr[offper])

            if r_on == 0 and r_off == 0:
                bias = np.nan
            else:
                bias = (r_on-r_off)/(np.abs(r_on)+np.abs(r_off))

            plt.suptitle(f'{exp_name}\n{stimname}'
                         f'\n{clusterids[i]} Rating: {clusters[i][2]}\n')

            if fr.max() < 20:
                bias = np.nan

            onoffbias[i] = bias
            baselines[i] = baseline

            all_frs.append(fr)

            ax2 = plt.subplot(212)
            plt.plot(t, fr)
            for eachslice in [onper, offper]:
                ax2.fill_between(t[eachslice], fr[eachslice],
                                 baseline, where=fr[eachslice] > baseline,
                                 facecolor='lightgray')

            plf.spineless(ax2)
            plt.axis([0, total_cycle, fr.min(), fr.max()])

            plt.title(f'Baseline: {baseline:2.0f} Hz Bias: {bias:0.2f}')
            plt.xlabel('Time[s]')
            plt.ylabel('Firing rate[spikes/s]')

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

        keystosave = ['clusters', 'total_cycle', 'bins', 'tstep',
                      'stimname', 'stim_duration', 'preframe_duration',
                      'contrast', 'all_frs', 't', 'exp_name', 'onoffbias',
                      'baselines']
        data_in_dict = {}
        for key in keystosave:
            data_in_dict[key] = locals()[key]

        np.savez(os.path.join(savedir, stim_nr + '_data'), **data_in_dict)
        print(f'Analysis of {stimname} completed.')

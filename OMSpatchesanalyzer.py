#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 18:13:17 2018

@author: ycan
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import analysis_scripts as asc
import plotfuncs as plf
import iofuncs as iof

def OMSpatchesanalyzer(exp_name, stim_nrs):
    """
    Analyze and plot the responses to object motion patches stimulus.
    """

    exp_dir = iof.exp_dir_fixer(exp_name)

    exp_name = os.path.split(exp_dir)[-1]

    if len(stim_nrs) == 0:
        return

    if isinstance(stim_nrs, int):
        stim_nrs = [stim_nrs]

    clusters, metadata = asc.read_ods(exp_dir, cutoff=4)
    clusterids = plf.clusters_to_ids(clusters)
    all_omsi = np.empty((clusters.shape[0], len(stim_nrs)))
    stimnames = []
    for stim_index, stim_nr in enumerate(stim_nrs):
        stim_nr = str(stim_nr)

        stimname = iof.getstimname(exp_dir, stim_nr)
        stimnames.append(stimname)

        parameters = asc.read_parameters(exp_dir, stim_nr)

        refresh_rate = metadata['refresh_rate']

        nblinks = asc.parameter_dict_get(parameters, 'Nblinks', 1)
        seed = asc.parameter_dict_get(parameters, 'seed', -10000)
        stim_duration = asc.parameter_dict_get(parameters, 'stimFrames',
                                               1400)
        # The duration in the parameters refers to the total duration of both
        # epochs. We divide by two to get the length of a single stim_durations
        stim_duration = int(stim_duration/2)
        prefr_duration = asc.parameter_dict_get(parameters, 'preFrames',
                                                100)

        frametimings = asc.readframetimes(exp_dir, stim_nr)

        # ntrials is the number of trials containing both
        ntrials = np.rint((frametimings.shape[0] / (stim_duration+1)))/2
        ntrials = ntrials.astype(int)
#        ntrials = int((frametimings.shape[0] / (stim_duration+1))/2)
        frametimings_rs = frametimings[:ntrials*2*(stim_duration+1)]
        frametimings_rs = frametimings_rs.reshape((ntrials*2, stim_duration+1))

        ft_local = frametimings_rs[::2][:, :-1]
        ft_global = frametimings_rs[1::2][:, :-1]

        localspikes = np.empty((clusters.shape[0], ntrials, stim_duration))
        globalspikes = np.empty((clusters.shape[0], ntrials, stim_duration))

        for i, cluster in enumerate(clusters):
            spikes = asc.read_raster(exp_name, stim_nr, cluster[0],
                                     cluster[1])
            for j in range(ntrials):
                localspikes[i, j, :] = asc.binspikes(spikes, ft_local[j, :])
                globalspikes[i, j, :] = asc.binspikes(spikes, ft_global[j, :])

        response_local = localspikes.mean(axis=1)
        response_global = globalspikes.mean(axis=1)

        # Differential and coherent firing rates
        fr_d = response_local.mean(axis=1)
        fr_c = response_global.mean(axis=1)

        # Calculate object motion sensitivity index (OMSI) as defined in
        # Kühn et al, 2016
        # There the first second of each trial is discarded, here it does not
        # seem to be very different from the rest.
        omsi = (fr_d - fr_c) / (fr_d + fr_c)

        savepath = os.path.join(exp_dir, 'data_analysis', stimname)
        if not os.path.isdir(savepath):
            os.makedirs(savepath, exist_ok=True)

        for i, cluster in enumerate(clusters):
            ax = plt.subplot(111)
            ax.plot(response_local[i, :], label='Local')
            ax.plot(response_global[i, :], label='Global')
            ax.set_title(f'{exp_name}\n{stimname}\n'
                         f'{clusterids[i]} OMSI: {omsi[i]:4.2f}')
            ax.set_xlabel('Time [ms]')
            ax.set_ylabel('Average firing rate [au]')
            plf.spineless(ax, 'tr')
            plt.savefig(os.path.join(savepath, clusterids[i]+'.svg'),
                        bbox_inches='tight')
            ax.legend(fontsize='x-small')
            plt.close()

        keystosave = ['nblinks', 'refresh_rate', 'stim_duration',
                      'prefr_duration', 'ntrials', 'response_local',
                      'response_global', 'fr_d', 'fr_c', 'omsi']
        datadict = {}

        for key in keystosave:
            datadict[key] = locals()[key]

        npzfpath = os.path.join(savepath, str(stim_nr)+'_data')
        np.savez(npzfpath, **datadict)
        all_omsi[:, stim_index] = omsi

    # Draw the distribution of the OMSI for all OMSI stimuli
    # If there is only one OMS stimulus, draw it in the same folder
    # If there are multiple stimuli, save it in the data analysis folder
    if len(stim_nrs) == 1:
        pop_plot_savepath = os.path.join(savepath, 'omsi_population.svg')
    else:
        pop_plot_savepath = os.path.split(savepath)[0]
        pop_plot_savepath = os.path.join(pop_plot_savepath, 'all_omsi.svg')

    ax2 = plt.subplot(111)
    for j, stim_nr in enumerate(stim_nrs):
        np.random.seed(j)
        ax2.scatter(all_omsi[:, j], j + np.random.random(omsi.shape)/10)
    np.random.seed()
    ax2.set_yticks(np.arange(len(stim_nrs)))
    ax2.set_yticklabels(stimnames, fontsize='xx-small', rotation='45')
    ax2.set_xlabel('Object-motion sensitivity index')
    ax2.set_title(f'{exp_name}\nDistribution of OMSI')
    plf.spineless(ax2, 'tr')
    plt.savefig(pop_plot_savepath)
    plt.show()

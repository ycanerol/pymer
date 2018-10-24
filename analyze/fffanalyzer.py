#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 15:21:50 2018

@author: ycan
"""
import matplotlib.pyplot as plt
import numpy as np
import os

from ..external import randpy
from ..modules import analysisfuncs as asc
from ..modules import iofuncs as iof
from ..plot import util as plf


def fff(exp_name, stimnrs):
    """
    Analyzes and plots data from full field flicker
    stimulus.
    """
    exp_dir = iof.exp_dir_fixer(exp_name)
    exp_name = os.path.split(exp_dir)[-1]

    if isinstance(stimnrs, int):
        stimnrs = [stimnrs]

    for stimnr in stimnrs:
        stimnr = str(stimnr)

        stimname = iof.getstimname(exp_name, stimnr)

        clusters, metadata = asc.read_spikesheet(exp_dir)

        parameters = asc.read_parameters(exp_dir, stimnr)

        clusterids = plf.clusters_to_ids(clusters)

        refresh_rate = metadata['refresh_rate']

        if parameters['stixelheight'] < 600 or parameters['stixelwidth'] < 800:
            raise ValueError('Make sure the stimulus is full field flicker.')

        nblinks = parameters['Nblinks']

        bw = parameters.get('blackwhite', False)

        seed = parameters.get('seed', -10000)

        filter_length, frametimings = asc.ft_nblinks(exp_dir, stimnr,
                                                     nblinks, refresh_rate)

        frame_duration = np.average(np.ediff1d(frametimings))
        total_frames = frametimings.shape[0]

        all_spiketimes = []
        # Store spike triggered averages in a list containing correct shaped
        # arrays
        stas = []
        # Make a list for covariances of the spike triggered ensemble
        covars = []
        for i in range(len(clusters[:, 0])):
            spiketimes = asc.read_raster(exp_dir, stimnr,
                                         clusters[i, 0], clusters[i, 1])
            spikes = asc.binspikes(spiketimes, frametimings)
            all_spiketimes.append(spikes)
            stas.append(np.zeros(filter_length))
            covars.append(np.zeros((filter_length, filter_length)))

        if bw:
            randnrs, seed = randpy.ranb(seed, total_frames)
            # Since ranb returns zeros and ones, we need to convert the zeros
            # into -1s.
            stimulus = np.array(randnrs) * 2 - 1
        else:
            randnrs, seed = randpy.gasdev(seed, total_frames)
            stimulus = np.array(randnrs)

        for k in range(filter_length, total_frames-filter_length+1):
            stim_small = stimulus[k-filter_length+1:k+1][::-1]
            for j in range(clusters.shape[0]):
                spikes = all_spiketimes[j]
                if spikes[k] != 0:
                    stas[j] += spikes[k]*stim_small
                    # This trick is needed to use .T for tranposing
                    stim_small_n = stim_small[np.newaxis, :]
                    # Calculate the covariance as the weighted outer product
                    # of small stimulus(i.e. snippet) with itself
                    # This is non-centered STC (a la Cantrell et al., 2010)
                    covars[j] += spikes[k]*(np.dot(stim_small_n.T,
                                                   stim_small_n))
        spikenrs = np.array([spikearr.sum() for spikearr in all_spiketimes])

        plotpath = os.path.join(exp_dir, 'data_analysis',
                                stimname, 'filters')
        if not os.path.isdir(plotpath):
            os.makedirs(plotpath, exist_ok=True)

        t = np.arange(filter_length)*frame_duration*1000

        eigvals = [np.zeros((filter_length)) for i in range(clusters.shape[0])]
        eigvecs = [np.zeros((filter_length,
                             filter_length)) for i in range(clusters.shape[0])]

        for i in range(clusters.shape[0]):
            stas[i] = stas[i]/spikenrs[i]
            covars[i] = covars[i]/spikenrs[i]
            eigvals[i], eigvecs[i] = np.linalg.eigh(covars[i])
            fig = plt.figure(figsize=(9, 6))
            ax = plt.subplot(111)
            ax.plot(t, stas[i], label='STA')
            ax.plot(t, eigvecs[i][:, 0], label='STC component 1', alpha=.5)
            ax.plot(t, eigvecs[i][:, -1], label='STC component 2', alpha=.5)
            # Add eigenvalues as inset
            ax2 = fig.add_axes([.65, .15, .2, .2])
            # Highlight the first and second components which are plotted
            ax2.plot(0, eigvals[i][0], 'o',
                     markersize=7, markerfacecolor='C1', markeredgewidth=0)
            ax2.plot(filter_length-1, eigvals[i][-1], 'o',
                     markersize=7, markerfacecolor='C2', markeredgewidth=0)
            ax2.plot(eigvals[i], 'ko', alpha=.5, markersize=4,
                     markeredgewidth=0)
            ax2.set_axis_off()
            plf.spineless(ax)
            ax.set_xlabel('Time[ms]')
            ax.set_title(f'{exp_name}\n{stimname}\n{clusterids[i]} Rating:'
                         f' {clusters[i, 2]} {int(spikenrs[i])} spikes')
            plt.savefig(os.path.join(plotpath, clusterids[i])+'.svg',
                        format='svg', dpi=300)
            plt.close()

        savepath = os.path.join(os.path.split(plotpath)[0], stimnr+'_data')

        keystosave = ['stas', 'clusters', 'frame_duration', 'all_spiketimes',
                      'stimname', 'total_frames', 'spikenrs', 'bw', 'nblinks',
                      'filter_length', 'exp_name', 'covars', 'eigvals',
                      'eigvecs']
        data_in_dict = {}
        for key in keystosave:
            data_in_dict[key] = locals()[key]

        np.savez(savepath, **data_in_dict)
        print(f'Analysis of {stimname} completed.')

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 14:06:26 2018

@author: ycan
"""
import matplotlib.pyplot as plt
import numpy as np
import os

from ..external import randpy
from ..modules import iofuncs as iof
from ..modules import analysisfuncs as asc
from ..plot import util as plf


def saccadegratings(exp_name, stim_nr):
    """
    Analyze and save responses to saccadegratings stimulus.
    """

    exp_dir = iof.exp_dir_fixer(exp_name)
    exp_name = os.path.split(exp_dir)[-1]
    stimname = iof.getstimname(exp_dir, stim_nr)
    clusters, metadata = asc.read_spikesheet(exp_dir)
    clusterids = plf.clusters_to_ids(clusters)

    refresh_rate = metadata['refresh_rate']

    parameters = asc.read_parameters(exp_name, stim_nr)
    if parameters['stimulus_type'] != 'saccadegrating':
        raise ValueError('Unexpected stimulus type: '
                         f'{parameters["stimulus_type"]}')
    fixfr = parameters.get('fixationframes', 80)
    sacfr = parameters.get('saccadeframes', 10)
    barwidth = parameters.get('barwidth', 40)
    averageshift = parameters.get('averageshift', 2)
    # The seed is hard-coded in the Stimulator
    seed = -10000

    ftimes = asc.readframetimes(exp_dir, stim_nr)
    ftimes.resize(int(ftimes.shape[0]/2), 2)
    nfr = ftimes.size
    # Re-generate the stimulus
    # Amplitude of the shift and the transition type (saccade or grey is
    # determined based on the output of ran1
    randnrs = np.array(randpy.ran1(seed, nfr)[0])

    # Separate the amplitude and transitions into two arrays
    stimpos = (4*randnrs[::2]).astype(int)

    # Transition variable, determines whether grating is moving during
    # the transion or only a grey screen is presented.
    trans = np.array(randnrs[1::2] > 0.5)

    # Record before and after positions in a single array and remove
    # The first element b/c there is no before value
    stimposx = np.append(0, stimpos)[:-1]
    stimtr = np.stack((stimposx, stimpos), axis=1)[1:]
    trans = trans[:-1]

    saccadetr = stimtr[trans, :]
    greytr = stimtr[~trans, :]

    # Create a time vector with defined temporal bin size
    tstep = 0.01  # Bin size is defined here, unit is seconds
    trialduration = (fixfr + sacfr)/refresh_rate
    nrsteps = int(trialduration/tstep)+1
    t = np.linspace(0, trialduration, num=nrsteps)

    # Collect saccade beginning time for each trial
    trials = ftimes[1:, 0]
    sacftimes = trials[trans]
    greyftimes = trials[~trans]

    sacspikes = np.empty((clusters.shape[0], sacftimes.shape[0],
                          t.shape[0]))
    greyspikes = np.empty((clusters.shape[0], greyftimes.shape[0],
                           t.shape[0]))
    # Collect all the psth in one array. The order is
    # transision type, cluster index, start pos, target pos, time
    psth = np.zeros((2, clusters.shape[0], 4, 4, t.size))

    for i, (chid, clid, _) in enumerate(clusters):
        spiketimes = asc.read_raster(exp_dir, stim_nr, chid, clid)
        for j, _ in enumerate(sacftimes):
            sacspikes[i, j, :] = asc.binspikes(spiketimes,
                                               sacftimes[j]+t)
        for k, _ in enumerate(greyftimes):
            greyspikes[i, k, :] = asc.binspikes(spiketimes,
                                                greyftimes[k]+t)

    # Sort trials according to the transition type
    # nton[i][j] contains the indexes of trials where saccade was i to j
    nton_sac = [[[] for _ in range(4)] for _ in range(4)]
    for i, trial in enumerate(saccadetr):
        nton_sac[trial[0]][trial[1]].append(i)
    nton_grey = [[[] for _ in range(4)] for _ in range(4)]
    for i, trial in enumerate(greytr):
        nton_grey[trial[0]][trial[1]].append(i)

    savedir = os.path.join(exp_dir, 'data_analysis', stimname)
    os.makedirs(savedir, exist_ok=True)
    for i in range(clusters.shape[0]):
        fig, axes = plt.subplots(4, 4, sharex=True, sharey=True,
                                 figsize=(8, 8))
        for j in range(4):
            for k in range(4):
                # Start from bottom left corner
                ax = axes[3-j][k]
                # Average all transitions of one type
                psth_sac = sacspikes[i, nton_sac[j][k], :].mean(axis=0)
                psth_grey = greyspikes[i, nton_grey[j][k], :].mean(axis=0)
                # Convert to spikes per second
                psth_sac = psth_sac/tstep
                psth_grey = psth_grey/tstep
                psth[0, i, j, k, :] = psth_sac
                psth[1, i, j, k, :] = psth_grey
                ax.axvline(sacfr/refresh_rate*1000, color='red',
                           linestyle='dashed', linewidth=.5)
                ax.plot(t*1000, psth_sac, label='Saccadic trans.')
                ax.plot(t*1000, psth_grey, label='Grey trans.')
                ax.set_yticks([])
                ax.set_xticks([])
                # Cosmetics
                plf.spineless(ax)
                if j == k:
                    ax.set_facecolor((1, 1, 0, 0.15))
                if j == 0:
                    ax.set_xlabel(f'{k}')
                    if k == 3:
                        ax.legend(fontsize='xx-small', loc=0)
                if k == 0:
                    ax.set_ylabel(f'{j}')

        # Add an encompassing label for starting and target positions
        ax0 = fig.add_axes([0.08, 0.08, .86, .86])
        plf.spineless(ax0)
        ax0.patch.set_alpha(0)
        ax0.set_xticks([])
        ax0.set_yticks([])
        ax0.set_ylabel('Start position')
        ax0.set_xlabel('Target position')
        plt.suptitle(f'{exp_name}\n{stimname}\n{clusterids[i]}')
        plt.savefig(os.path.join(savedir, f'{clusterids[i]}.svg'))
        plt.close()
    # Save results
    keystosave = ['fixfr', 'sacfr', 't', 'averageshift', 'barwidth', 'seed',
                  'trans', 'saccadetr', 'greytr', 'nton_sac', 'nton_grey',
                  'stimname', 'sacspikes', 'greyspikes', 'psth', 'nfr',
                  'parameters']
    data_in_dict = {}
    for key in keystosave:
        data_in_dict[key] = locals()[key]

    np.savez(os.path.join(savedir, str(stim_nr) + '_data'), **data_in_dict)
    print(f'Analysis of {stimname} completed.')

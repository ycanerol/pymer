#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 16 09:47:24 2018

@author: ycan
"""
import matplotlib.pyplot as plt
import numpy as np
import os

from .. import io as iof
from ..modules import analysisfuncs as asc
from ..plot import util as plf


def allfff(exp_name, stim_nrs):
    """
    Plot all of the full field flicker STAs on top of each other
    to see the progression of the cell responses, their firing rates.
    """

    if isinstance(stim_nrs, int) or len(stim_nrs) <= 1:
        print('Multiple full field flicker stimuli expected, '
              'allfff analysis will be skipped.')
        return

    exp_dir = iof.exp_dir_fixer(exp_name)
    exp_name = os.path.split(exp_dir)[-1]

    # Sanity check to ensure we are commparing the same stimuli and parameters
    prev_parameters = {}
    for i in stim_nrs:
        pars = asc.read_parameters(exp_name, i)
        currentfname = pars.pop('filename')
        if len(prev_parameters) == 0:
            prev_parameters = pars
        for k1, k2 in zip(pars.keys(), prev_parameters.keys()):
            if pars[k1] != prev_parameters[k2]:
                raise ValueError(
                    f'Parameters for {currentfname} do not match!\n'
                    f'{k1}:{pars[k1]}\n{k2}:{prev_parameters[k2]}')

    stimnames = []
    for j, stim in enumerate(stim_nrs):
        data = iof.load(exp_name, stim)
        stas = data['stas']
        clusters = data['clusters']
        filter_length = data['filter_length']
        frame_duration = data['frame_duration']
        if j == 0:
            all_stas = np.zeros((clusters.shape[0], filter_length,
                                len(stim_nrs)))
            all_spikenrs = np.zeros((clusters.shape[0], len(stim_nrs)))
        all_stas[:, :, j] = stas
        all_spikenrs[:, j] = data['spikenrs']
        stimnames.append(iof.getstimname(exp_name, stim))

    t = np.linspace(0, frame_duration*filter_length, num=filter_length)
    # %%
    clusterids = plf.clusters_to_ids(clusters)
    for i in range(clusters.shape[0]):
        fig = plt.figure()
        ax1 = plt.subplot(111)
        ax1.plot(t, all_stas[i, :, :])
        ax1.set_xlabel('Time [ms]')
        ax1.legend(stimnames, fontsize='x-small')
        ax2 = fig.add_axes([.65, .15, .2, .2])
        for j in range(len(stim_nrs)):
            ax2.plot(j, all_spikenrs[i, j], 'o')
        ax2.set_ylabel('# spikes', fontsize='small')
        ax2.set_xticks([])
        ax2.patch.set_alpha(0)
        plf.spineless(ax1, 'tr')
        plf.spineless(ax2, 'tr')
        plt.suptitle(f'{exp_name}\n {clusterids[i]}')
        plotpath = os.path.join(exp_dir, 'data_analysis', 'all_fff')
        if not os.path.isdir(plotpath):
            os.makedirs(plotpath, exist_ok=True)
        plt.savefig(os.path.join(plotpath, clusterids[i])+'.svg',
                    format='svg', dpi=300)
        plt.close()
    print('Plotted full field flicker STAs together from all stimuli.')

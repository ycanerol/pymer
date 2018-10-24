#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 11:20:35 2018

@author: ycan
"""
import matplotlib.pyplot as plt
import numpy as np
import os
import warnings

from .. import io as iof
from ..plot import util as plf


def allonoff(exp_name, stim_nrs):

    if isinstance(stim_nrs, int) or len(stim_nrs) <= 1:
        print('Multiple onoffsteps stimuli expected, '
              'allonoff analysis will be skipped.')
        return

    exp_dir = iof.exp_dir_fixer(exp_name)
    exp_name = os.path.split(exp_dir)[-1]

    for j, stim in enumerate(stim_nrs):
        data = iof.load(exp_name, stim)
        all_frs = data['all_frs']
        clusters = data['clusters']
        preframe_duration = data['preframe_duration']
        stim_duration = data['stim_duration']
        onoffbias = data['onoffbias']
        t = data['t']

        if j == 0:
            a = np.zeros((clusters.shape[0], t.shape[0], len(stim_nrs)))
            bias = np.zeros((clusters.shape[0], len(stim_nrs)))
        a[:, :, j] = np.array(all_frs)
        bias[:, j] = onoffbias

    plotpath = os.path.join(exp_dir, 'data_analysis', 'allonoff')
    clusterids = plf.clusters_to_ids(clusters)
    if not os.path.isdir(plotpath):
        os.makedirs(plotpath, exist_ok=True)

    for i in range(clusters.shape[0]):
        ax = plt.subplot(111)
        for j, stim in enumerate(stim_nrs):
            labeltxt = (iof.getstimname(exp_name,
                                        stim).replace('onoffsteps_', '')
                        + f' Bias: {bias[i, j]:4.2f}')
            plt.plot(t, a[i, :, j], alpha=.5,
                     label=labeltxt)
        plt.title(f'{exp_name}\n{clusterids[i]}')
        plt.legend()
        plf.spineless(ax)
        plf.drawonoff(ax, preframe_duration, stim_duration, h=.1)

        plt.savefig(os.path.join(plotpath, clusterids[i])+'.svg',
                    format='svg', dpi=300)
        plt.close()

    rows = len(stim_nrs)
    columns = 1
    _, axes = plt.subplots(rows, columns, sharex=True)
    colors = plt.get_cmap('tab10')

    for i, stim in enumerate(stim_nrs):
        ax = axes[i]
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            ax.hist(bias[:, i], bins=20, color=colors(i),
                    range=[-1, 1], alpha=.5)

        ax.set_ylabel(iof.getstimname(exp_name,
                                      stim).replace('onoffsteps_', ''))
        plf.spineless(ax)
    plt.suptitle(f'Distribution of On-Off Indices for {exp_name}')
    plt.subplots_adjust(top=.95)
    plt.xlabel('On-Off index')
    plt.savefig(os.path.join(exp_dir, 'data_analysis', 'onoffindex_dist.svg'),
                format='svg', dpi=300)
    plt.close()

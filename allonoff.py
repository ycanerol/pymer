#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 11:20:35 2018

@author: ycan
"""
import iofuncs as iof
import numpy as np
import matplotlib.pyplot as plt
import plotfuncs as plf
import os


#exp_name = '20180124'
#onoffsteps = [3, 8, 14]

def allonoff(exp_name, stim_nrs):

    if isinstance(stim_nrs, int):
        raise ValueError('Multiple stimuli should be given!')

    exp_dir = iof.exp_dir_fixer(exp_name)
    exp_name = os.path.split(exp_dir)[-1]


    for i, stim in enumerate(stim_nrs):
        data = iof.load(exp_name, stim)
        all_frs = data['all_frs']
        clusters = data['clusters']
        preframe_duration = data['preframe_duration']
        stim_duration = data['stim_duration']
        t = data['t']

        if i == 0:
            a = np.zeros((clusters.shape[0], t.shape[0], len(stim_nrs)))
        a[:, :, i] = np.array(all_frs)

    plotpath = os.path.join(exp_dir, 'data_analysis', 'allonoff')
    clusterids = plf.clusters_to_ids(clusters)
    if not os.path.isdir(plotpath):
        os.makedirs(plotpath, exist_ok=True)

    for i in range(clusters.shape[0]):
        ax = plt.subplot(111)
        for j in range(len(stim_nrs)):
            plt.plot(t, a[i, :, j], alpha = .5,
                     label=iof.getstimname(exp_name, stim_nrs[j]))
        plt.title(f'{exp_name}\n{clusterids[i]}')
        plt.legend()
        plf.spineless(ax)
        plf.drawonoff(ax, preframe_duration, stim_duration, h=.1)

        plt.savefig(os.path.join(plotpath, clusterids[i])+'.svg',
                    format='svg', dpi=300)
        plt.close()

allonoff('20180124',)

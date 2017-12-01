#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 11:35:22 2017

@author: ycan
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import analysis_scripts as asc
import plotfuncs as plf

experiment_dir = '/home/ycan/Documents/data/Erol_20171122_252MEA_fr_re_fp'

spont_stimuli = [1, 2]

wdir = os.getcwd()
try:
    os.chdir(experiment_dir)
    stim_fnames = [glob.glob('%s_*.mcd' % i) for i in spont_stimuli]
finally:
    os.chdir(wdir)
stim_fnames = sum(stim_fnames, [])  # Hacky solution to convert list of lists
stim_names = [s.split('.mcd')[0] for s in stim_fnames]
stim_nrs = [s.split('_')[0] for s in stim_names]

clusters, _ = asc.read_ods(experiment_dir, cutoff=3)

total_spikes = np.empty([len(stim_fnames), len(clusters[:, 0])])

for i in range(len(stim_fnames)):
    all_spikes = []
    interspike_int = []
    for j in range(len(clusters[:, 0])):
        spikes = asc.read_raster(experiment_dir, stim_nrs[i],
                                 clusters[j, 0], clusters[j, 1])
        all_spikes.append(spikes)
        total_spikes[i, j] = len(spikes)
        interspike_int.append(np.ediff1d(spikes))
    plt.eventplot(all_spikes)
    plt.ylabel('Cluster')
    plt.xlabel('Time[s]')
    plt.title('{} rasters'.format(stim_names[i]))
    plt.show()
#%%
    plt.figure()
    for i in range(len(clusters[:, 0])):
        plt.hist(interspike_int[i], bins=np.linspace(0,.050))
        plt.title('interspike intervals for cell {}'.format(i))
        plt.xlabel('Time between spikes[s]')
        plt.show()
#%%
plt.plot(total_spikes.T)
plt.legend(stim_names)
plt.title('Firing rate for each cell')
plt.xlabel('Cluster')
plt.ylabel('Number of spikes')
plt.show()

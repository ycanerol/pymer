#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 18:18:03 2017

@author: ycan
"""
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
import analysis_scripts as asc
import matplotlib as mpl

stim_duration = 60
preframe_duration = 120
# Convert from number of frames to seconds
stim_duration = stim_duration/60
preframe_duration = preframe_duration/60
total_cycle = (stim_duration+preframe_duration)*2
contrast = 1

experiment_dir = '/home/ycan/Documents/data/Erol_20171116_252MEA_sr_le_sp'

onoffstimulus_order = 3

wdir = os.getcwd()
try:
    os.chdir(experiment_dir)
    stimulusname = glob.glob('%s_*.mcd' % onoffstimulus_order)[0]
finally:
    os.chdir(wdir)

stim_names = stimulusname.split('.mcd')[0]
stim_nrs = stimulusname.split('_')[0]

clusters, _ = asc.read_ods(experiment_dir+'/spike_sorting.ods', cutoff=4)

frametimings, _ = asc.getframetimes(experiment_dir +
                                    '/RawChannels/{}_253.bin'.format(onoffstimulus_order))
frametimings = frametimings /1000
all_spikes = []
for i in range(len(clusters[:, 0])):
    spikes = asc.read_raster(experiment_dir, onoffstimulus_order,
                             clusters[i, 0], clusters[i, 1])
    all_spikes.append(spikes)
#%%
fig = plt.figure()
ax = plt.subplot(111)

rect1 = mpl.patches.Rectangle((stim_duration/total_cycle, 0),
                              width=preframe_duration/total_cycle, height=1,
                              transform=ax.transAxes, color='k',
                              alpha=.5)
rect2 = mpl.patches.Rectangle(((preframe_duration+stim_duration)/total_cycle, 0),
                              width=stim_duration/total_cycle, height=1,
                              transform=ax.transAxes, color='k',
                              alpha=1)
rect3 = mpl.patches.Rectangle(((preframe_duration+2*stim_duration)/total_cycle, 0),
                              width=preframe_duration/total_cycle, height=1,
                              transform=ax.transAxes, color='k',
                              alpha=.5)
ax.add_patch(rect1)
ax.add_patch(rect2)
ax.add_patch(rect3)
plt.show()

frametimings.resize((np.ceil(frametimings.shape[0]/4).astype(int)), 4)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 14:06:26 2018

@author: ycan
"""
import numpy as np
import matplotlib.pyplot as plt

import iofuncs as iof
import analysis_scripts as asc
import plotfuncs as plf
from randpy import randpy


exp_name = '20180710'
stim_nr = 7

def saccadegratingsanalyzer(exp_name, stim_nr):
    pass

exp_dir = iof.exp_dir_fixer(exp_name)
stimname = iof.getstimname(exp_dir, stim_nr)
clusters, metadata = asc.read_spikesheet(exp_dir)

refresh_rate = metadata['refresh_rate']

parameters = asc.read_parameters(exp_name, stim_nr)
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
trans = trans[1:]

saccadetr = stimtr[trans, :]
greytr = stimtr[~trans, :]

# Create a time vector with defined temporal bin size
tstep = 0.01  # Bin size is defined here, unit is seconds
trialduration = (fixfr + sacfr)/refresh_rate
nrsteps = int(trialduration/tstep)+1
t = np.linspace(0, trialduration, num=nrsteps)

# Collect saccade beginning, saccade end and fixation end for each trial
trials = np.concatenate((ftimes[:-1, :], ftimes[1:, 0][:, None]), axis=1)
sacftimes = trials[trans, :]
greyftimes = trials[~trans, :]

sacspikes = np.empty((clusters.shape[0], sacftimes.shape[0], t.shape[0]))
greyspikes = np.empty((clusters.shape[0], greyftimes.shape[0], t.shape[0]))

for i, (chid, clid, _) in enumerate(clusters):
    spiketimes = asc.read_raster(exp_dir, stim_nr, chid, clid)
    for j, trial in enumerate(sacftimes):
        sacspikes[i, j, :] = asc.binspikes(spiketimes, sacftimes[j, 0]+t)
    for k, greytrial in enumerate(greyftimes):
        greyspikes[i, k, :] = asc.binspikes(spiketimes, greyftimes[k, 0]+t)

# Sort trials according to the transition type
# nton[i][j] contains the indexes of trials where saccade was i to j
nton_sac = [[[] for _ in range(4)] for _ in range(4)]
for i, trial in enumerate(saccadetr):
    nton_sac[trial[0]][trial[1]].append(i)
nton_grey = [[[] for _ in range(4)] for _ in range(4)]
for i, trial in enumerate(greytr):
    nton_grey[trial[0]][trial[1]].append(i)

#%%
for i in range(clusters.shape[0]):
    fig, axes = plt.subplots(4, 4, sharex=True, sharey=True, figsize=(8, 8))
    for j in range(4):
        for k in range(4):
            ax = axes[j][k]
            # Average all transitions of one type
            psth_sac = sacspikes[i, nton_sac[j][k], :].mean(axis=0)
            psth_grey = greyspikes[i, nton_grey[j][k], :].mean(axis=0)
            # Convert to spikes per second
            psth_sac = psth_sac/tstep
            psth_grey = psth_grey/tstep
            ax.axvline(sacfr/refresh_rate, color='red',
                       linestyle='dashed')
            ax.plot(t, psth_sac, label='Saccadic trans.')
            ax.plot(t, psth_grey, label='Grey trans.')
            plf.spineless(ax)
            if j == k:
                ax.set_facecolor((1, 1, 0, 0.15))
            if j == 3: ax.set_xlabel(f'{k}')
            if k == 0: ax.set_ylabel(f'{j}')
    plt.show()
    break
#    if i == 5: break
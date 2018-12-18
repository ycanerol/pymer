#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 12:15:52 2018

@author: ycan
"""
#%%
import numpy as np
import analysis_scripts as asc
from randpy import randpy

# fix the total number of frames to be loaded to avoid running out of memory
total_frames_fix = 10000

def loadstim(exp, stim_nr, maxframenr=10000):
    sortedstim = asc.stimulisorter(exp)
    clusters, metadata = asc.read_spikesheet(exp)
    pars = asc.read_parameters(exp, stim_nr)

    for key, val in sortedstim.items():
        if stim_nr in val:
            stimtype = key
    if stimtype in ['fff', 'stripeflicker', 'checkerflicker']:
        seed = pars.get('seed', -10000)
        bw = pars.get('blackwhite', False)
        filter_length, frametimings = asc.ft_nblinks(exp, stim_nr)
        total_frames = frametimings.shape[0]

        if stimtype == 'fff':
            if bw:
                randnrs, seed = randpy.ranb(seed, total_frames)
                # Since ranb returns zeros and ones, we need to convert
                # the zeros into -1s.
                stimulus = np.array(randnrs) * 2 - 1
            else:
                randnrs, seed = randpy.gasdev(seed, total_frames)
                stimulus = np.array(randnrs)
        elif stimtype == 'checkerflicker':
            scr_width = metadata['screen_width']
            scr_height = metadata['screen_height']
            stx_h = pars['stixelheight']
            stx_w = pars['stixelwidth']
            # Check whether any parameters are given for margins, calculate
            # screen dimensions.
            marginkeys = ['tmargin', 'bmargin', 'rmargin', 'lmargin']
            margins = []
            for key in marginkeys:
                margins.append(pars.get(key, 0))
            # Subtract bottom and top from vertical dimension; left and right
            # from horizontal dimension
            scr_width = scr_width-sum(margins[2:])
            scr_height = scr_height-sum(margins[:2])
            sx, sy = scr_height/stx_h, scr_width/stx_w
            # Make sure that the number of stimulus pixels are integers
            # Rounding down is also possible but might require
            # other considerations.
            if sx % 1 == 0 and sy % 1 == 0:
                sx, sy = int(sx), int(sy)
            else:
                raise ValueError('sx and sy must be integers')

            # HINT: fixing stimulus lengt for now because of memory capacity
            total_frames = maxframenr

            randnrs, seed = randpy.ranb(seed, sx*sy*total_frames)
            # Reshape and change 0's to -1's
            stimulus = np.reshape(randnrs, (sx, sy, total_frames), order='F')*2-1
        return stimulus
    if stimtype == 'OMB':
        stimframes = pars.get('stimFrames', 108000)
        preframes = pars.get('preFrames', 200)
        nblinks = pars.get('Nblinks', 2)

        seed = pars.get('seed', -10000)
        seed2 = pars.get('objseed', -1000)

        stepsize = pars.get('stepsize', 2)

        ntotal = int(stimframes / nblinks)

        randnrs, seed = randpy.gasdev(seed, ntotal*2)
        randnrs = np.array(randnrs)*stepsize

        xsteps = randnrs[::2]
        ysteps = randnrs[1::2]

        return np.vstack((xsteps, ysteps))
    return None
#%%

def calc_ste(spikes, stimulus, filter_length):
    stimshape = stimulus.shape[:-1] # Stimulus shape excluding the temporal dim
    ste = np.zeros((spikes[filter_length:].sum(), # Number of rows for the STE
                    *stimshape, # Based on the nr of non-temporal dimensions of the stimulus
                    filter_length)) # The temporal dimension
    st_i = 0
    spikes_reduced=[]
    for i, spike in enumerate(spikes):
        if i < filter_length-1: continue
        if spike == 0: continue
        for _ in range(spike):
            ste[st_i, ...] = stimulus[..., i-filter_length+1:i+1][...,::-1]
            spikes_reduced.append(spike)
            st_i += 1
    return ste, np.array(spikes_reduced)
# %% Load test data for full field flicker
import iofuncs as iof

exp = '20180710'
stimnr = asc.stimulisorter(exp)['fff'][0]
data = iof.load(exp, stimnr)
filter_length = data['filter_length']


cell_i = 6
spikes = data['all_spiketimes'][cell_i]
ste, spikes = calc_ste(spikes, loadstim(exp, stimnr), filter_length)

import matplotlib.pyplot as plt
plt.plot(ste.mean(axis=0))
plt.plot(data['stas'][cell_i])
plt.show()

#%%
# We exclude some spikes in the beginning of the stimulus

sta = np.matmul(ste.T, spikes)
stc = np.linalg.inv(np.matmul(ste.T, ste))

# Center STC
sta_large = sta.repeat(ste.shape[0]).reshape(ste.shape)
ste_centered = ste - sta_large
stc_centered = np.linalg.inv(np.matmul(ste_centered.T, ste_centered))
del ste_centered
k = np.matmul(stc, sta)

#%% Test data for checkerflicker

cell_i = 5
exp = '20180207'
stimnr = asc.stimulisorter(exp)['checkerflicker'][0]
frnr = total_frames_fix
data = iof.load(exp, stimnr)
spikes = data['all_spiketimes'][cell_i]
filter_length = data['filter_length']

stechecker = calc_ste(spikes[:frnr], loadstim(exp, stimnr), filter_length)

sta = data['stas'][cell_i]
max_i = data['max_inds'][cell_i]
maxframe = sta[:, :, max_i[-1]]
#%%
plt.subplot(121)
plt.imshow(maxframe)
plt.subplot(122)
plt.imshow(stechecker.mean(axis=0)[..., max_i[-1]])

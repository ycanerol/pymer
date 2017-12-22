#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 11:22:39 2017

@author: ycan
"""

import lnp_checkerflicker as lnp
import analysis_scripts as asc
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import randpy
import datetime
import glob
import os
exp_dir = '/home/ycan/Documents/data/Erol_20171116_252MEA_sr_le_sp/'

stimulusnr = 6
cutoff = 3

stimname = os.path.split(glob.glob(exp_dir+'%s_*.mcd' % stimulusnr)[-1])[-1]
stimname = stimname.split('.mcd')[0]

clusters, metadata = asc.read_ods(exp_dir, cutoff=cutoff)

scr_width, scr_height = metadata['screen_width'], metadata['screen_height']

parameters = asc.read_parameters(exp_dir, stimulusnr)

stx_h = parameters['stixelheight']
stx_w = parameters['stixelwidth']

# Check whether any parameters are given for margins, calculate
# screen dimensions.
marginkeys = ['tmargin', 'bmargin', 'rmargin', 'lmargin']
margins = []
for key in marginkeys:
    try:
        margins.append(parameters[key])
    except KeyError:
        margins.append(0)

# Subtract bottom and top from vertical dimension; left and right
# from horizontal dimension
scr_width = scr_width-sum(margins[2:])
scr_height = scr_height-sum(margins[:2])

nblinks = parameters['Nblinks']
try:
    seed = parameters['seed']
except KeyError:
    seed = -10000

sx, sy = scr_height/stx_h, scr_width/stx_w

# Make sure that the number of stimulus pixels are integers
# Rounding down is also possible but might require
# other considerations.
if sx % 1 == 0 and sy % 1 == 0:
    sx, sy = int(sx), int(sy)
else:
    raise ValueError('sx and sy must be integers')

# %%
# If the frame rate of the checkerflicker stimulus is 16 ms, (i.e.
# Nblinks is set to 1), frame timings should be handled differently
# because the state of the pulse can only be changed when a new
# frame is delivered. For this reason, the offsets of the pulses
# also denote a frame change as well as onsets.
if nblinks == 1:
    ft_on, ft_off = asc.getframetimes(exp_dir, stimulusnr,
                                      returnoffsets=True)
    # Initialize empty array twice the size of one of them, assign
    # value from on or off to every other element.
    frametimings = np.empty(ft_on.shape[0]*2, dtype=float)
    frametimings[::2] = ft_on
    frametimings[1::2] = ft_off
    # Set filter length so that temporal filter is ~600 ms. The unit
    # here is number of frames.
    filter_length = 40
elif nblinks == 2:
    frametimings = asc.getframetimes(exp_dir, stimulusnr)
    filter_length = 20
#%%
#######
# Try with few cells and short recording
#clusters = clusters[:, :]
#frametimings = frametimings[:30000]
########
frame_duration = np.average(np.ediff1d(frametimings))
total_frames = frametimings.shape[0]

# %%
all_spiketimes = []
# Store spike triggered averages in a list containing correct shaped arrays
stas = []
# Store number of spikes during the calculation to use in the averaging
spikenrs = np.zeros(clusters.shape[0]).astype('int')
for i in range(len(clusters[:, 0])):
    spiketimes = asc.read_raster(exp_dir, stimulusnr,
                                 clusters[i, 0], clusters[i, 1])

    spikes = asc.binspikes(spiketimes, frametimings)
    all_spiketimes.append(spikes)
    stas.append(np.zeros((sx, sy, filter_length)))
# %%
# Length of the chunks (specified in number of frames)
chunklength = 10000
chunksize = chunklength*sx*sy
nrofchunks = int(np.ceil(total_frames/chunklength))
time = startime = datetime.datetime.now()
for i in range(nrofchunks):
    randnrs, seed = randpy.ran1(seed, chunksize)
    randnrs = [1 if i > .5 else -1 for i in randnrs]
    stimulus = np.reshape(randnrs, (sx, sy, chunklength), order='F')
    del randnrs
    # Range of indices we are interested in for the current chunk
    if (i+1)*chunklength < total_frames:
        chunkind = slice(i*chunklength, (i+1)*chunklength)
        chunkend = chunklength
    else:
        chunkind = slice(i*chunklength, None)
        chunkend = total_frames - i*chunklength
#    frames_small = frametimings[chunkind]
    for k in range(filter_length, chunkend-filter_length+1):
        stim_small = stimulus[:, :, k-filter_length+1:k+1][:, :, ::-1]
        for j in range(clusters.shape[0]):
            spikes = all_spiketimes[j][chunkind]
            if spikes[k] != 0:
                stas[j] += spikes[k]*stim_small
                spikenrs[j] += spikes[k]
#            frames_small_nz = frames_small[spikes > 0]
#            spikes = spikes[spikes > 0]
#            snippets = np.dot(frames_small_nz, spikes)
    print('Chunk {:>2} completed in {}'.format(i, datetime.datetime.now()-time))
    time = datetime.datetime.now()
# %%
max_inds = []
for i in range(clusters.shape[0]):
    stas[i] = stas[i]/np.sum(all_spiketimes[i])
    max_inds.append(np.squeeze(np.where(np.abs(stas[i]) == np.max(np.abs(stas[i])))))

print('Total elapsed time: {}'.format(datetime.datetime.now()-startime))

savepath = os.path.join(exp_dir, 'data_analysis', stimname)
if not os.path.isdir(savepath):
    os.makedirs(savepath, exist_ok=True)
np.savez(savepath+'/data.npz',
         clusters=clusters,
         frametimings=frametimings,
         all_spiketimes=all_spiketimes,
         frame_duration=frame_duration,
         max_inds=max_inds,
         nblinks=nblinks,
         stas=stas,
         stx_h=stx_h,
         stx_w=stx_w,
         total_frames=total_frames)
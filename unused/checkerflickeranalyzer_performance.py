#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 11:22:39 2017

@author: ycan
"""

import numpy as np
import randpy
import datetime
from os.path import join as pjoin
import os
import analysis_scripts as asc
import iofuncs as iof
import miscfuncs as msc
import matplotlib.pyplot as plt
import plotfuncs as plf


def norm(a):
    x = np.ravel(a)
    return x / np.sqrt(np.sum(np.power(x, 2)))


def selfcorr(stas, original_stas):
    cors = np.array([])
    for i in range(len(stas)):
        cor = np.correlate(norm(stas[i]), norm(original_stas[i]))
        cors = np.append(cors, cor)
    return cors

#%%
def zscore(sta):
    z = (np.max(np.abs(sta)) - sta.mean()) / sta.std()
    return z
#%%
#
#exp_name = '20171122'
#stimulusnr = 7
#cutoff = 1
exp_dir = iof.exp_dir_fixer(exp_name)
original_stas = iof.loadh5(pjoin(exp_dir, 'data_analysis',
                                 iof.getstimname(exp_name, stimulusnr),
                                 str(stimulusnr)+'_data.h5'))
original_stas = original_stas['stas']
desired_chunknr = 50
"""
Measures approximation quality in each chunk to the original sta

Parameters:
----------
    exp_name:
        Experiment name.
    stimulusnr:
        Number of the stimulus to be analyzed.
    cutoff:

"""

stimname = iof.getstimname(exp_name, stimulusnr)

exp_name = os.path.split(exp_dir)[-1]

clusters, metadata = asc.read_ods(exp_dir, cutoff=cutoff)

scr_width = metadata['screen_width']
scr_height = metadata['screen_height']

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

# If the frame rate of the checkerflicker stimulus is 16 ms, (i.e.
# Nblinks is set to 1), frame timings should be handled differently
# because the state of the pulse can only be changed when a new
# frame is delivered. For this reason, the offsets of the pulses
# also denote a frame change as well as onsets.
if nblinks == 1:
    ft_on, ft_off = asc.readframetimes(exp_dir, stimulusnr,
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
    frametimings = asc.readframetimes(exp_dir, stimulusnr)
    filter_length = 20
elif nblinks == 4:
    frametimings = asc.readframetimes(exp_dir, stimulusnr)
    # There are two pulses per frame
    frametimings = frametimings[::2]
    filter_length = 10
else:
    raise ValueError('nblinks is expected to be 1, 2 or 4.')

savefname = str(stimulusnr)+'_performance'

frame_duration = np.average(np.ediff1d(frametimings))
total_frames = frametimings.shape[0]

all_spiketimes = []
# Store spike triggered averages in a list containing correct shaped
# arrays
stas = []

for i in range(len(clusters[:, 0])):
    spiketimes = asc.read_raster(exp_dir, stimulusnr,
                                 clusters[i, 0], clusters[i, 1])

    spikes = asc.binspikes(spiketimes, frametimings)
    all_spiketimes.append(spikes)
    stas.append(np.zeros((sx, sy, filter_length)))


# Empirically determined to be best for 32GB RAM
desired_chunk_size = 216000000
nrofchunks = 1
while desired_chunknr > nrofchunks:
    # Length of the chunks (specified in number of frames)
    if nrofchunks != 1:
        desired_chunk_size /= 2
    chunklength = int(desired_chunk_size/(sx*sy))

    chunksize = chunklength*sx*sy
    nrofchunks = int(np.ceil(total_frames/chunklength))

print('Chunk length :{} frames\n'
      'Total nr of chunks: {}'.format(chunklength, nrofchunks))
time = startime = datetime.datetime.now()

corrs = np.empty(len(stas))

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

    for k in range(filter_length, chunkend-filter_length+1):
        stim_small = stimulus[:, :, k-filter_length+1:k+1][:, :, ::-1]
        for j in range(clusters.shape[0]):
            spikes = all_spiketimes[j][chunkind]
            if spikes[k] != 0:
                stas[j] += spikes[k]*stim_small
    corrs = np.vstack((corrs, selfcorr(stas, original_stas)))
    print('Chunk {:>2} out of {} completed'
          ' in {}'.format(i+1, nrofchunks, msc.timediff(time)))
    time = datetime.datetime.now()

max_inds = []
spikenrs = np.array([spikearr.sum() for spikearr in all_spiketimes])

for i in range(clusters.shape[0]):
    stas[i] = stas[i]/spikenrs[i]
    # Find the pixel with largest absolute value
    max_i = np.squeeze(np.where(np.abs(stas[i])
                                == np.max(np.abs(stas[i]))))
    # If there are multiple pixels with largest value,
    # take the first one.
    if max_i.shape != (3,):
        max_i = max_i[:, 0]
    max_inds.append(max_i)

print('Total elapsed'
      ' time: {}'.format(msc.timediff(startime)))

corrs = corrs[1:, ...]
t = (np.arange(nrofchunks)*chunklength*frame_duration)/60
#%%
savepath = os.path.join(exp_dir, 'data_analysis', stimname)
if not os.path.isdir(savepath):
    os.makedirs(savepath, exist_ok=True)
savepath = os.path.join(savepath, savefname)

ax = plt.subplot(111)
ax.plot(t, corrs, alpha=0.3)
plt.ylabel('Similarity to final STA')
plt.xlabel('Minutes of stimulus analyzed')
plt.ylim([0, 1])
plf.spineless(ax, 'tr')
plt.title('Recording duration optimization\n{}\n {}'.format(exp_name, savefname))
plt.savefig(savepath+'.svg', format='svg')

#%%

#    with h5py.File(savepath+'.h5', mode='w') as f:
#        keystosave = ['clusters', 'frametimings', 'all_spiketimes',
#                      'frame_duration', 'max_inds', 'nblinks', 'stas',
#                      'stx_h', 'stx_w', 'total_frames', 'sx', 'sy',
#                      'filter_length', 'stimname', 'exp_name', 'spikenrs']
#        lists = []
#        for key in keystosave:
#            f[key] = locals()[key]
#
np.savez(savepath,
         corrs=corrs,
         t=t,
         nrofchunks=nrofchunks,
         chunklength=chunklength,
         frame_duration=frame_duration,
         total_frames=total_frames,
         filter_length=filter_length,
         stimname=stimname,
         exp_name=exp_name,
         spikenrs=spikenrs)

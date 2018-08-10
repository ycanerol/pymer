#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 11:43:06 2018

@author: ycan
"""

exp_name = '20180207'

stim_nr = 12

path = '/home/ycan/Documents/data/Erol_20180207/data_analysis/12_stripeflicker5bw1blink_highl_corrected/12_data.npz'

import os
import iofuncs as iof
import matplotlib.pyplot as plt
import plotfuncs as plf
import numpy as np
import analysis_scripts as asc

exp_dir = iof.exp_dir_fixer(exp_name)

_, metadata = asc.read_spikesheet(exp_dir)
px_size = metadata['pixel_size(um)']

data = {}
with np.load(path) as f:
    # Get all the variable names that were saved
    keys = list(f.keys())
    for key in keys:
        item = f[key]
        if item.shape == ():
            item = np.asscalar(item)
        data[key] = item
# Some variables (e.g. STAs) are stored as lists originally
# but saving to and loading from npz file converts them to
# numpy arrays with one additional dimension. To revert
# this, we need to turn them back into lists with list()
# function. The variables that should be converted are
# to be kept in list_items.
list_of_lists = iof.list_of_lists
list_items = list_of_lists

for list_item in list_items:
    if list_item in keys:
        data[list_item] = list(data[list_item])

clusters = data['clusters']
stas = data['stas']
filter_length = data['filter_length']
stx_w = data['stx_w']
exp_name = data['exp_name']
stimname = data['stimname']

stimname += '_corrected'

frame_duration = data['frame_duration']
quals = data['quals']

clusterids = plf.clusters_to_ids(clusters)

# Determine frame size so that the total frame covers
# an area large enough i.e. 2*700um
t = np.arange(filter_length)*frame_duration*1000
vscale = int(stas[0].shape[0] * stx_w*px_size/1000)
for i in range(clusters.shape[0]):
    sta = stas[i]

    vmax = np.max(np.abs(sta))
    vmin = -vmax
    plt.figure(figsize=(6, 15))
    ax = plt.subplot(111)
    im = ax.imshow(sta, cmap='RdBu', vmin=vmin, vmax=vmax,
                   extent=[0, t[-1], -vscale, vscale], aspect='auto')
    plt.xlabel('Time [ms]')
    plt.ylabel('Distance [mm]')

    plf.spineless(ax)
    plf.colorbar(im, ticks=[vmin, 0, vmax], format='%.2f', size='2%')
    plt.suptitle(f'{exp_name}\n{stimname}\n'
                 f'{clusterids[i]} Rating: {clusters[i][2]}\n'
                 f'STA quality: {quals[i]:4.2f}')
    plt.subplots_adjust(top=.90)
    savepath = os.path.join(exp_dir, 'data_analysis',
                            stimname, 'STAs')
    if not os.path.isdir(savepath):
        os.makedirs(savepath, exist_ok=True)
    plt.savefig(os.path.join(savepath, clusterids[i]+'.svg'),
                bbox_inches='tight')
    plt.close()
print(f'Plotting of {stimname} completed.')

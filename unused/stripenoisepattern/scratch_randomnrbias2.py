#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 18:20:40 2018

@author: ycan
"""
from randpy import randpy
import numpy as np
import matplotlib.pyplot as plt
import plotfuncs as plf

sy = 160
total_frames = 77510
filter_length = 40
seed = -1000

sta = np.zeros((sy, filter_length))
spikect = 0

# Initialize the stimulus
randnrs, seed = randpy.ran1(seed, sy*filter_length)
randnrs = [1 if i > .5 else -1 for i in randnrs]
stim = np.reshape(randnrs, (sy, filter_length), order='F')

for frame in range(total_frames):
    randnrs, seed = randpy.ran1(seed, sy)
    randnrs = [1 if i > .5 else -1 for i in randnrs]
    stim = np.hstack((stim[:, 1:], np.array(randnrs)[..., None]))
    spike = np.random.poisson()
    spike = 1
    if spike != 0:
        sta += stim*spike
        spikect += spike
sta /= spikect
# %%
plt.figure(figsize=(6, 14))
plf.stashow(sta, plt.gca())
plt.show()

#%%
bar = sta[:, -1]
clusters = data['clusters']
stas = data['stas']
clusterids = plf.clusters_to_ids(clusters)

rows = 1
columns = 2

for i in range(clusters.shape[0]):
    orig_sta = stas[i]

    plt.figure(figsize=(14, 14))
    plt.title(clusterids[i])
    realsta = np.hstack((orig_sta, np.zeros(shape=(sy, 3)), bar[..., None]))
    ax1 = plt.subplot(rows, columns, 1)
    plf.stashow(orig_sta, ax1)

    #col = np.mean(realsta[:, 25:-5], axis=1)
    #ax2 = plt.subplot(rows, columns, 3)
    #ax2.plot(col, label='real')
    #ax2.plot(bar, label='random artifact')


    bars = np.hstack((bar.reshape((sy, 1)))*filter_length)
    ax3 = plt.subplot(rows, columns, 2)
    bars = np.outer(bar, np.ones(filter_length))
    sta_cor = orig_sta - bars
    plt.figure(figsize=(6, 14))
    plf.stashow(sta_cor, ax3)
    plt.close()
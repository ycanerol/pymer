#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""
#%%

import numpy as np
import matplotlib.pyplot as plt

import iofuncs as iof
import analysis_scripts as asc
import plotfuncs as plf

from omb import OMB
import scratch_matchOMBandchecker as moc

exp, ombstimnr = 'Kuehn', 13
checkerstimnr = 1


st = OMB(exp, ombstimnr,
         maxframes=10000
         )

choosecells = [54, 55, 108, 109]
nrcells = len(choosecells)

ombcoords = np.zeros((nrcells, 2))
all_spikes = np.zeros((nrcells, st.ntotal), dtype=np.int8)
#all_contrasts = np.zeros((nrcells, st.ntotal))

for i, cell in enumerate(choosecells):
    ombcoords[i, :] = moc.chkmax2ombcoord(cell, exp, ombstimnr, checkerstimnr)
    all_spikes[i, :] = st.binnedspiketimes(cell)
#    all_contrasts[i, :] = st.generatecontrast(ombcoords[i, :])

contrast = st.generatecontrast(st.texpars.noiselim/2, 100).astype(np.float32)
contrast_sum = asc.normalize(contrast.sum(axis=2), axis_inv=None)
plt.imshow(contrast_sum, cmap='Greys_r')
#%%
rw = asc.rolling_window(contrast, 20)
#rws = rw.transpose((2, 0, 1, 3))

stas = np.einsum('abcd,ec->eabd', rw, all_spikes)
stas = stas / all_spikes.sum(axis=1)[:, None, None, None]
stas_normalized = asc.normalize(stas, axis_inv=0)
plt.imshow(stas[0, ..., 0], cmap='Greys_r')
#%%
from scratch_spikeshuffler import shufflebyrow

shuffled_spikes = shufflebyrow(all_spikes)

shuffled_stas = np.einsum('abcd,ec->eabd', rw, shuffled_spikes)
shuffled_stas = shuffled_stas / all_spikes.sum(axis=1)[:, None, None, None]
shuffled_stas_normalized = asc.normalize(shuffled_stas, axis_inv=0)
plt.imshow(shuffled_stas[0, ..., 0], cmap='Greys_r')
#%%
times_to_shuffle = 10

all_shuffled_stas = np.empty((times_to_shuffle, *stas.shape))

for i in range(times_to_shuffle):
    all_shuffled_stas[i, ...] = np.einsum('abcd,ec->eabd',
                                           rw,
                                           shufflebyrow(all_spikes))
all_shuffled_stas = all_shuffled_stas / all_spikes.sum(axis=1)[None, :, None, None, None]


#%%
# Match the stimulus screen exactly
# make the Qt window full screen on second monitor

plt.imshow(contrast[..., 0], cmap='Greys_r')
plt.axis('off')
plt.subplots_adjust(top=0.915,
                    bottom=0.108,
                    left=0.016,
                    right=0.984,
                    hspace=0.2,
                    wspace=0.2)
plt.show()

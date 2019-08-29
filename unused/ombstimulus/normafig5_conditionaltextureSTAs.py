#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Recreate Fig 5A and 5B from Kühn and Gollisch 2019

Conditional texture STAs for preferred and non-preferred directions

"""

import numpy as np
import matplotlib.pyplot as plt

import iofuncs as iof
import analysis_scripts as asc
import plotfuncs as plf

from omb import OMB
import nonlinearity as nlt

def subtract_avgcontrast(sta, contrast_avg):
    """
    Subtract the average contrast from the STAs

    Example
    -------
    >>> sta.shape # (ncells, xpix, ypix, filter_length)
    (4, 61, 61, 20)
    >>> contrast.shape # (xpix, ypix, ntotal)
    (61, 61, 20)
    >>> contrast_avg = contrast.mean(axis=-1)
    >>> sta_corrected = subtract_avgcontrast(stas, contrast_avg)
    """
    return sta - contrast_avg[None, :, :, None]


exp, ombstimnr = 'Kuehn', 13
checkerstimnr = 1


st = OMB(exp, ombstimnr,
         maxframes=1000
         )

choosecells = [54, 55, 108, 109]
nrcells = len(choosecells)

all_spikes = np.zeros((nrcells, st.ntotal), dtype=np.int8)

for i, cell in enumerate(choosecells):
    all_spikes[i, :] = st.binnedspiketimes(cell)

rw = asc.rolling_window(st.bgsteps, st.filter_length)

motionstas = np.einsum('abc,db->dac', rw, all_spikes)
motionstas /= all_spikes.sum(axis=(-1))[:, np.newaxis, np.newaxis]

#%% Filter the stimuli

# Euclidian norm
motionstas_norm = motionstas / np.sqrt((motionstas**2).sum(axis=-1))[:, :, None]

bgsteps = st.bgsteps / np.sqrt(st.bgsteps.var())
rw = asc.rolling_window(bgsteps, st.filter_length)


steps_proj = np.einsum('abc,bdc->ad', motionstas_norm, rw)

nbins = 15
bins = np.zeros((nrcells, nbins))
nlts = np.zeros((bins.shape))

for i in range(nrcells):
    nlts[i, :], bins[i, :]  = nlt.calc_nonlin(all_spikes[i, :], steps_proj[i, :], nr_bins=nbins)

nlts *= st.refresh_rate

#%%
pd = steps_proj > 0.5
npd = steps_proj < -0.5

pd_spikes = all_spikes.copy()
npd_spikes = all_spikes.copy()

# Exclude spikes that are not in desired direction to zero
pd_spikes[~pd] = 0
npd_spikes[~npd] = 0

#%%
filter_length = 20
contrast = st.generatecontrast([100, 100], 100, filter_length-1) # subtract one because rw generates one extra
#contrast = st.generatecontrastmaxi([0, 0], 800, filter_length-1) # subtract one because rw generates one extra
contrast_avg = contrast[:, :, filter_length+1:].mean(axis=-1)
rw = asc.rolling_window(contrast, filter_length, preserve_dim=False)
#%%
stas = np.einsum('abcd,ec->eabd', rw, all_spikes)
stas = stas / all_spikes.sum(axis=1)[:, None, None, None]

from scratch_spikeshuffler import shufflebyrow

shuffled_spikes = shufflebyrow(all_spikes)

shuffled_stas = np.einsum('abcd,ec->eabd', rw, shuffled_spikes)
shuffled_stas = shuffled_stas / shuffled_spikes.sum(axis=1)[:, None, None, None]

pdstas = np.einsum('abcd,ec->eabd', rw, pd_spikes)
pdstas = pdstas / pd_spikes.sum(axis=1)[:, None, None, None]

npdstas = np.einsum('abcd,ec->eabd', rw, npd_spikes)
npdstas = npdstas / npd_spikes.sum(axis=1)[:, None, None, None]

stas = subtract_avgcontrast(stas, contrast_avg)
shuffled_stas = subtract_avgcontrast(shuffled_stas, contrast_avg)
pdstas = subtract_avgcontrast(pdstas, contrast_avg)
npdstas = subtract_avgcontrast(npdstas, contrast_avg)
#%%
fig, axes = plt.subplots(2, 4)
ax1sta = axes[0, 0]
ax1sta.plot(motionstas[0, :, :].T)
ax1sta.set_title('Norma cell 55 (fig5atop)')

ax2sta = axes[1, 0]
ax2sta.set_title('Norma cell 109 (fig5abottom)')
ax2sta.plot(motionstas[2, :, :].T)

for i, cell in enumerate([0, 2]): # cell 55 and 109
    axnlt = axes[i, 1]
    axnlt.plot(bins[cell, :], nlts[cell, :])
    axnlt.spines['left'].set_position('center')

    axnpd = axes[i, 2]
    axpd = axes[i, 3]
    axnpd.imshow(npdstas[cell, ..., -6], cmap='Greys_r')
    axpd.imshow(pdstas[cell, ..., -6], cmap='Greys_r')
    axnpd.get_shared_x_axes().join(axnpd, axpd)
    axnpd.get_shared_y_axes().join(axnpd, axpd)

    axnpd.axis([60, 140, 60, 140])
    axpd.axis([60, 140, 60, 140])

    if i == 0:
        axnpd.set_title('Non-preferred dir')
        axpd.set_title('Preferred dir')

#plt.savefig('/home/ycan/Documents/meeting_notes/2019-06-19/normafig.pdf')

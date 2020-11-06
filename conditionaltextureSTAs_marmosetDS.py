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


exp, ombstimnr = '20180710_kilosorted', 8
checkerstimnr = 6


st = OMB(exp, ombstimnr,
         maxframes=None
         )

choosecells = [8, 33, 61, 73, 79]

nrcells = len(choosecells)

all_spikes = st.allspikes()[choosecells, :]

rw = asc.rolling_window(st.bgsteps, st.filter_length)

motionstas = np.array(st.read_datafile()['stas'])[choosecells, :]
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
texture_file = st.read_texture_analysis()
# Normally the texture stas are contrst avg subtracted
# it will bedone again here, so we add it
stas = texture_file['texturestas'][choosecells, :] + texture_file['contrast_avg'][None, ..., None]

from scratch_spikeshuffler import shufflebyrow

shuffled_spikes = shufflebyrow(all_spikes)

print('Calculating shuffled_stas')
shuffled_stas = np.einsum('abcd,ec->eabd', rw, shuffled_spikes)
shuffled_stas = shuffled_stas / shuffled_spikes.sum(axis=1)[:, None, None, None]

print('Calculating pdstas')
pdstas = np.einsum('abcd,ec->eabd', rw, pd_spikes)
pdstas = pdstas / pd_spikes.sum(axis=1)[:, None, None, None]

print('Calculating npdstas')
npdstas = np.einsum('abcd,ec->eabd', rw, npd_spikes)
npdstas = npdstas / npd_spikes.sum(axis=1)[:, None, None, None]

print('Subtracting avg ')
stas = subtract_avgcontrast(stas, contrast_avg)
shuffled_stas = subtract_avgcontrast(shuffled_stas, contrast_avg)
pdstas = subtract_avgcontrast(pdstas, contrast_avg)
npdstas = subtract_avgcontrast(npdstas, contrast_avg)
#%%
fig, axes = plt.subplots(nrcells, 4, figsize=(15, 8))

for i, cell in enumerate(choosecells):
    axsta = axes[i, 0]
    axsta.plot(motionstas[i, :, :].T)
    axsta.set_title(f'Cell {st.clids[cell]}')

    axnlt = axes[i, 1]
    axnlt.plot(bins[i, :], nlts[i, :])
    axnlt.spines['left'].set_position('center')

    axnpd = axes[i, 2]
    axpd = axes[i, 3]
    axnpd.imshow(npdstas[i, ..., -6], cmap='Greys_r')
    axpd.imshow(pdstas[i, ..., -6], cmap='Greys_r')
    axnpd.get_shared_x_axes().join(axnpd, axpd)
    axnpd.get_shared_y_axes().join(axnpd, axpd)

    texture_sta_frame = [50, 150, 50, 150]
    axnpd.axis(texture_sta_frame)
    axpd.axis(texture_sta_frame)

    if i == 0:
        axnpd.set_title('Non-preferred dir')
        axpd.set_title('Preferred dir')

plt.savefig(f'/home/ycan/Downloads/2020-05-25_labmeeting/conditionalSTAs_{st.ntotal}.pdf')

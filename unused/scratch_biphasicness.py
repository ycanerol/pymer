#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""

import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import find_peaks

from stimulus import Stimulus

def biphasic_index2(sta):
    posarea = np.trapz(sta[sta > 0])
    negarea = np.trapz(sta[sta < 0])
    phasic_index = np.abs(posarea + negarea) / (np.abs(negarea) + np.abs(posarea)) # Ravi et al., 2019 J.Neurosci
    biphasic_index = 1 - phasic_index
    return biphasic_index

def biphasic_index3(sta):
    return np.abs(sta.max() + sta.min()) / (np.abs(sta.max()) + np.abs(sta.min()))

def biphasic_index3_stas(stas):
    maxs = stas.max(axis=1)
    mins = stas.min(axis=1)
    return 1 - (np.abs(maxs + mins) / (np.abs(maxs) + np.abs(mins)))

ff = Stimulus('20180710', 1)
ff = Stimulus('Kuehn', 2)

data = ff.read_datafile()
stas = np.array(data['stas'])

bps = np.empty(ff.nclusters)
bps2 = np.empty(ff.nclusters)
bps3 = biphasic_index3_stas(stas)

#%%
for i in range(ff.nclusters):

    pospeaks = find_peaks(stas[i, :], prominence=.2)[0]
    negpeaks = find_peaks(-stas[i, :], prominence=.2)[0]
    peaks = np.sort(np.hstack((pospeaks, negpeaks)))
    plt.plot(stas[i, :])
    plt.plot(peaks, stas[i, peaks], 'ro')
    if peaks.shape[0] == 2:
        bps[i] = np.abs(stas[i, peaks[1]] / stas[i, peaks[0]])
    else:
        bps[i] = np.nan
    bps2[i] = biphasic_index2(stas[i, :])
    plt.title(f'cell {i} ,bps: {bps[i]:4.2f} bps2: {bps2[i]:4.2f}, bps3:{bps3[i]:4.2f}')
    plt.show()
plt.hist(bps)
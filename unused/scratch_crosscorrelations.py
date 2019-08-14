#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""

import numpy as np
import matplotlib.pyplot as plt

import pycorrelate

import iofuncs as iof
import analysis_scripts as asc

exp = '20180710'
stim = 8

data = iof.load(exp, stim)
clusters = data['clusters']

allspikes = data['all_spikes']

#plt.xcorr(allspikes[0, :])

def corr(x1, x2=None, window=200):
    if x2 is None:
        x2 = x1
    assert x1.shape == x2.shape
    mid = x1.shape[0]
    # This is super slow,
    out = np.correlate(x1, x2, 'full')[mid-window-1:mid+window]
    return out

def corr2(x1, x2=None, window=0.05, res=0.001):
    if x2 is None:
        x2 = x1
    corr = pycorrelate.pcorrelate(x1, x2, window, res)
    return corr
#%%
def corr3(x1, x2=None, window=50):
    """
    x1 and x2 are binned spikes.
    """
    if x2 is None:
        x2 = x1
    corr = np.zeros((window*2+1))
    # Positive lag
    corr[:window+1] = pycorrelate.ucorrelate(x1, x2, window+1)[::-1]
    # Negative lag
    corr[window:] = pycorrelate.ucorrelate(x2, x1, window+1)

    return corr


def compare(i, j, allspikes, window=50):
    x1 = allspikes[i, :]
    x2 = allspikes[j, :]
    a = corr(x1, x2, window=window)
    b = corr3(x1, x2, window=window)
    plt.plot(a, label='corr')
    plt.plot(b, label='corr3')
    plt.legend()
    plt.show()
#
#compare(0, 0, allspikes)
#compare(1, 0, allspikes)
#compare(0, 1, allspikes)

def corr4(i1, i2, window, bin_length=0.001):
    """
    Correlation between spike trains. Accepts spike rasters.

    Parameters
    --------
    i1, i2:
        Indices of spike rasters to compare
    window:
        The maximum time lag to consider, in seconds.
    bin_length:
        Length of each bin. Default is 1 ms.
    """
    i1 = clusters[i1][:2]
    i2 = clusters[i2][:2]
    x1 = asc.read_raster(exp, stim, *i1)
    x2 = asc.read_raster(exp, stim, *i2)
    corr_bins = np.arange(-window, window, bin_length)
    corr = pycorrelate.pcorrelate(x1, x2, corr_bins, normalize=True)
    return corr




#%%
def allcorr2(allspikes, window):
    pxc = np.zeros((n, n, window))
    for i in range(n):
        for j in range(n):
            if i >= j:
                pxc[i, j, :] = pycorrelate.ucorrelate(allspikes[i, :], allspikes[j, :], window)

#%%
plt.close()
n = len(clusters)
window = 50
xcorrs = np.zeros((n, n, window*2+1))

spcorwindow, spcorbin = 0.05, 0.001
spcorrs = np.zeros((n, n, int(spcorwindow/spcorbin)*2-1))
#fig, axes = plt.subplots(n, n)
for i in range(n):
    for j in range(n):
#
#        ax = axes[i, j]
#        ax = plt.subplot(n, n, n*i+j+1)
#        ax.set_axis_off()
        if i >= j:
            xcorrs[i, j, :] = corr3(allspikes[i, :], allspikes[j, :], window)
            spcorrs[i, j, :] = corr4(i, j, spcorwindow, spcorbin)
#            ax.plot(xcorrs[i, j, :], lw=.5)
#plt.show()

#%%
from scipy import signal

def plotpeaks(i, j):
    xcorr = xcorrs[i, j, :]
    peaks = signal.find_peaks(xcorr, prominence=200)[0]
    plt.plot(xcorr)
    plt.plot(peaks, xcorr[peaks], 'x')
    plt.axvline(xcorr.shape[0]/2, color='grey', linestyle='dashed')
    plt.show()

plotpeaks(10, 1)
#%%
spikes = asc.read_raster(exp, stim, 1, 1)
spikes2 = asc.read_raster(exp, stim, 1, 3)

t = np.arange(0, spikes[-1], 0.001)
bsp = asc.binspikes(spikes, t)
bsp2 = asc.binspikes(spikes2, t)

#plt.plot(corr(bsp, window=40))

sprcorwindow = 0.05
corr_bin = np.arange(-sprcorwindow, sprcorwindow, .001)

pycorp = pycorrelate.pcorrelate(spikes, spikes2, corr_bin, normalize=True)
plt.plot(pycorp)
plt.show()

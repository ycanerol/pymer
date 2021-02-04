#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""

import numpy as np
import matplotlib.pyplot as plt
import pycorrelate

import time

import iofuncs as iof
import analysis_scripts as asc

exp = '20180710'
stim = 8

data = iof.load(exp, stim)

clusters = data['clusters']
allspikes = data['all_spikes']
frame_duration = data['frame_duration']


def corr(x1, x2=None, window=200):
    if x2 is None:
        x2 = x1
    assert x1.shape == x2.shape
    mid = x1.shape[0]
    # This is super slow,
    out = np.correlate(x1, x2, 'full')[mid-window-1:mid+window]
    return out


def corr_binned(x1, x2=None, window=50):
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


def corr_raster(i1, i2, window, bin_length=0.001):
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
    try:
        corr = pycorrelate.pcorrelate(x1, x2, corr_bins, normalize=True)
    except ZeroDivisionError:
        corr = np.full(corr_bins.shape[0]-1, np.nan)
    return corr


def all_xcorrs_binned(allspikes, window):
    n = allspikes.shape[0]
    xcorrs = np.zeros((n, n, window*2+1))
    xcorrs_norm = np.zeros((xcorrs.shape))
    for i in range(n):
        for j in range(n):
            xcorr = corr_binned(allspikes[i, :], allspikes[j, :], window)
            xcorrs[i, j, :] = xcorr
            xcorrnorm = xcorr-np.median(xcorr)
            xcorrnorm = xcorrnorm/xcorrnorm.max()
            xcorrs_norm[i, j, :] = xcorrnorm
    return xcorrs, xcorrs_norm


def all_xcorrs_raster(exp, stim, clusters, spcorwindow, spcorbinlen):
    n = len(clusters)
    spcorrs = np.zeros((n, n, int(spcorwindow/spcorbinlen)*2-1))
    spcorrs_norm = np.zeros(spcorrs.shape)
    for i in range(n):
        for j in range(n):
            if i >= j:
                corr = corr_raster(i, j, spcorwindow, spcorbinlen)
                spcorrs[i, j, :] = corr
                corr = corr-np.median(corr)
                corr = corr/corr.max()
                spcorrs_norm[i, j, :] = corr
    return spcorrs, spcorrs_norm


#%%
window = 30
window_ras = 0.05
binlen_ras = 0.001
start = time.time()
xcorrs, xcorrs_norm = all_xcorrs_binned(allspikes, window)
spcorrs, spcorrs_norm = all_xcorrs_raster(exp, stim, clusters, window_ras, binlen_ras)

elapsed = time.time()-start
print(f'{elapsed/60:6.1f} mins for calculating all cross-correlations')
#%%
plt.close()
n = clusters.shape[0]
xoffset_bin = window*frame_duration*2.5
xoffset_ras = window_ras*2.5
yoffset = 2.5


t_bin = np.linspace(-window*frame_duration, window*frame_duration, window*2+1)
t_ras = np.linspace(-window_ras, window_ras, int(window_ras/binlen_ras)*2-1)
#%%
start = time.time()
fig1 = plt.figure(1)
fig2 = plt.figure(2)
ax_bin = fig1.add_subplot(111)
ax_ras = fig2.add_subplot(111)
for i in range(n):
    for j in range(n):

        ax_bin.set_axis_off()
        if i >= j:
            # Start plotting from the top
            xo = (n-i)*xoffset_bin
            xo_ras = (n-i)*xoffset_ras
            yo = j*yoffset
            ax_bin.plot(xo + t_bin, yo + xcorrs_norm[i, j, :], lw=.1)
            ax_bin.vlines(t_bin[window]+xo, ymin=yo-yoffset/3, ymax=yo+yoffset/3, color='grey',
                          alpha=0.6, linestyle='dashed', lw=.1)
            ax_bin.text(xo, yo-yoffset/4, f'{i:2},{j:2}', alpha=.4, fontsize=1)

            ax_ras.plot(xo_ras+t_ras, yo + spcorrs_norm[i, j, :], lw=.1)
            ax_ras.vlines(t_ras[int(len(t_ras)/2)]+xo_ras, ymin=yo-yoffset/3, ymax=yo+yoffset/3, color='grey',
                          alpha=0.6, linestyle='dashed', lw=.1)
            ax_ras.text(xo_ras, yo-yoffset/4, f'{i:2},{j:2}', alpha=.4, fontsize=1)

fig1.savefig('/home/ycan/Downloads/binned_corrs.png', dpi=1200)
fig2.savefig('/home/ycan/Downloads/raster_corrs.png', dpi=1200)
plt.show()
elapsed = time.time()-start
print(f'{elapsed:6.1f} secs for plotting all cross-correlations')

#%%

np.savez('/media/ycan/datadrive/data/20180802_YE_252MEA_Marmoset_eye1_421/data_analysis/11_OMB_bg4x4corr8stdv_C150_Gsteps3stdv/11_xcorrs.npz',
         xcorrs_norm=xcorrs_norm, xcorrs=xcorrs, spcorrs=spcorrs, spcorrs_norm=spcorrs, frame_duration=frame_duration, spcorbin=spcorbin,
         spcorwindow=spcorwindow, t_bin=t_bin, t_ras=t_ras, bincorwindow=window, window_ras=window_ras, binlen_ras=binlen_ras,)


from scipy import signal


def plotpeaks(i, j):
    xcorr = xcorrs[i, j, :]
    peaks = signal.find_peaks(xcorr, prominence=200)[0]
    plt.plot(xcorr)
    plt.plot(peaks, xcorr[peaks], 'x')
    plt.axvline((xcorr.shape[0]-1)/2, color='grey', linestyle='dashed')
    plt.show()


plotpeaks(10, 10)
#%%

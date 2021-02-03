#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

maximum expected likelihood estimator for GQM

Park and Pillow, 2013, NIPS

"""
import numpy as np
from scipy import stats
import analysis_scripts as asc

from stimulus import Stimulus
from omb import OMB

from randpy import randpy


def calc_stc_loops(spikes, stimulus, filter_length):
    N = spikes.shape[0]
    stc = np.zeros((filter_length, filter_length))
    for i in range(filter_length, N):
        snippet = spikes[i] * stimulus[i-filter_length+1:i+1]
        snippet -= sta
        stc += np.outer(snippet, snippet)
    stc /= (spikes.sum()-1)
    return stc


def calc_stca(spikes, stimulus, filter_length):
    rw = asc.rolling_window(stimulus, filter_length, preserve_dim=True)
    sta = (spikes @ rw) / spikes.sum()
    # STA is not projected out like Equation 4 in Schwartz et al.2006,J.Vision
    precovar = (rw * spikes[:, None]) - sta
    stc = (precovar.T @ precovar) / (spikes.sum()-1)
    return sta, stc

#%%
def confidence_interval_2d(data, *args, **kwargs):
    assert data.ndim == 2
    lows, highs = np.zeros(data.shape[0]), np.zeros(data.shape[0])
    for i in range(data.shape[0]):
        lows[i], highs[i] = confidence_interval(data[i, :], *args, **kwargs)
    return lows, highs


def confidence_interval(data, significance_level=.95):
    low, high = stats.t.interval(significance_level, len(data)-1,
                                 loc=np.mean(data),
                                 scale=stats.sem(data))
    return low, high


def sigtest(spikes, stimulus, filter_length, ntest=100):
    all_v = np.zeros((filter_length, ntest))
    significant_components = []
    for i in range(ntest):
        shifted_spikes = np.roll(spikes, np.random.randint(spikes.shape[0]))
        r_sta, r_stc = calc_stca(shifted_spikes, stimulus, filter_length)
        # project out stuff

        r_v, r_w = np.linalg.eigh(r_stc)
        all_v[:, i] = r_v

    return all_v

#x = sigtest(spikes, stimulus, filter_length)
#x_min, x_max = confidence_interval_2d(x)
#%%
#ax = plt.gca()
#ax.fill_between(np.arange(filter_length),
#                 x.min(axis=1), x.max(axis=1),
#                 color='grey', alpha=.5)
#ax.fill_between(np.arange(filter_length), x_min, x_max,
#                color='red', alpha=.5)
#ax.plot(eigvals, 'ok')
#%%
exp, stimnr = '20180710', 1

ff = Stimulus(exp, stimnr)
stimulus = np.array(randpy.gasdev(-1000, ff.frametimings.shape[0])[0])

st = OMB(exp, 8)
ff = st
stimulus = st.bgsteps[0, :]

allspikes = ff.allspikes()

i = 0
spikes = allspikes[i, :]
filter_length = ff.filter_length

rw = asc.rolling_window(stimulus, filter_length, preserve_dim=True)
sta = (spikes @ rw) / spikes.sum()
#%%
# I am not projecting out the STA like Equation 4 in Schwartz et al.2006,J.Vision
precovar = (rw * spikes[:, None]) - sta
stc = (precovar.T @ precovar) / (spikes.sum()-1)
eigvals, eigvecs = np.linalg.eig(stc)
eigsort = np.argsort(eigvals)
eigvals, eigvecs = eigvals[eigsort], eigvecs[:, eigsort]

stc2 = calc_stc(spikes, stimulus, filter_length)

fig,axes = plt.subplots(2, 1)
axes[0].plot(eigvals, 'o')
ax1 = axes[1]
ax1.plot(sta)
ax1.plot(eigvecs[:, 0])
ax1.plot(eigvecs[:, -1])

#%%
sp_mean = spikes.mean()
stim_covar = np.cov(rw, rowvar=False)
term1 = stc + (2*(sta@sta.T))/sp_mean
bmel = np.linalg.inv(term1)@sta
cmel = (np.linalg.inv(stim_covar) - sp_mean*np.linalg.inv(term1))/2

fig2, axes2 = plt.subplots(2, 2)
axes2 = axes2.flat
axes2[0].plot(bmel)
axes2[0].plot(sta, ls='dashed', color='grey', label='STA')
axes2[1].imshow(cmel, cmap='seismic', vmin=asc.absmin(cmel), vmax=asc.absmax(cmel))

v, w = np.linalg.eigh(cmel);
eiginds = [0, -1]
axes2[2].plot(v, 'ko');
axes2[2].plot(v[eiginds[0]], 'o');
axes2[2].plot(v[eiginds[1]], 'o');
axes2[3].plot(w[:, eiginds])
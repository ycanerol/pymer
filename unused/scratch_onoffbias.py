#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 17:31:17 2018

@author: ycan

Used to develop onoff bias calculation
"""

import iofuncs as iof
import numpy as np
import matplotlib.pyplot as plt
import plotfuncs as plf
import os

exp_name = '20180124'
stimnr = 3

data = iof.load(exp_name, stimnr)

exp_dir = iof.exp_dir_fixer(exp_name)

clusters = data['clusters']
preframe_duration = data['preframe_duration']
stim_duration = data['stim_duration']
t = data['t']
tstep = data['tstep']
all_frs = data['all_frs']
exp_name = data['exp_name']
stimname = data['stimname']

clusterids = plf.clusters_to_ids(clusters)

onbegin = preframe_duration
onend = onbegin+stim_duration
offbegin = onend+preframe_duration
offend = offbegin+stim_duration

a = []
for i in [onbegin, onend, offbegin, offend]:
    yo = np.asscalar(np.where(np.abs(t-i) < tstep/2)[0][-1])
    a.append(yo)

prefs = []
# To exclude stimulus offset affecting the bias, use
# last 1 second of preframe period
cut_time = 1
for i in [onbegin-1, onbegin, offbegin-1, offbegin]:
    yo = np.asscalar(np.where(np.abs(t-i) < tstep/2)[0][-1])
    prefs.append(yo)

onper = slice(a[0], a[1])
offper = slice(a[2], a[3])

pref1 = slice(prefs[0], prefs[1])
pref2 = slice(prefs[2], prefs[3])

onoffbias = np.empty(clusters.shape[0])
baselines = np.empty(clusters.shape[0])

for i in range(clusters.shape[0]):
    fr = all_frs[i]

    prefr = np.append(fr[pref1], fr[pref2])
    baseline = np.median(np.round(prefr))

    fr_corr = fr - baseline

    r_on = np.sum(fr_corr[onper])
    r_off = np.sum(fr_corr[offper])

    bias = (r_on-r_off)/(np.abs(r_on)+np.abs(r_off))

    if fr.max() < 20:
        print(clusters[i, :2], fr.max())
        bias = np.nan

    onoffbias[i] = bias
    baselines[i] = baseline

#    savepath = os.path.join(exp_dir,'data_analysis', stimname)
#    data.update({'onoffbias':onoffbias, 'baselines':baselines})
#    np.savez(os.path.join(savepath, f'{stimnr}_data.npz'), **data)


    ax = plt.subplot(111)
    ax.plot(t, fr)
    plt.axhline(baseline)
    plf.drawonoff(ax, preframe_duration, stim_duration, h=.2)
    plf.spineless(ax)
    plt.axis([0, t[-1], fr.min()-5, fr.max()])
    plt.ylabel('Firing rate[spikes/s]')
    plt.xlabel('Time [s]')
    plt.title(f'{exp_name} {stimname}\n{clusterids[i]}'+
              f'\nBias: {bias:0.2f}\nBaseline: {baseline:4.2f}')
    plt.savefig(os.path.join(exp_dir, 'data_analysis',
                             clusterids[i]+'onoffbiascalculation.pdf'),
                bbox_inches='tight', figsize=(12, 10))
    plt.show()

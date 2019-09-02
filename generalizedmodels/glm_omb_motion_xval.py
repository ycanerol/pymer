#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""
import os

import numpy as np
import matplotlib.pyplot as plt
import datetime as dt

import iofuncs as iof
import analysis_scripts as asc
import nonlinearity as nlt
import genlinmod_multidimensional as glm
from train_test_split import train_test_split

from omb import OMB
from stimulus import Stimulus

#exp, stim_nr = '20180710', 8
exp, stim_nr  = 'Kuehn', 13
xval_splits = 10
xval_fraction = 1/xval_splits

st = OMB(exp, stim_nr)

glmlabel = 'GLM_motion_xval'

savepath = os.path.join(st.stim_dir, glmlabel)
os.makedirs(savepath, exist_ok=True)

omb_stas = np.array(st.read_datafile()['stas'])
texture_data = st.read_texture_analysis()

all_spikes = st.allspikes()

start = dt.datetime.now()

kall = np.zeros((st.nclusters, 2, xval_splits, st.filter_length))
muall = np.zeros((st.nclusters, xval_splits))

frs = np.zeros((all_spikes.shape[0], int(all_spikes.shape[-1]/xval_splits)))

cross_corrs = np.zeros((st.nclusters, xval_splits))

t = np.linspace(0, st.filter_length*st.frame_duration*1000, st.filter_length)
stimulus = st.bgsteps

plotlabels = ['Motion X', 'Motion Y']

for i, cluster in enumerate(st.clusters):
    for xvi in range(xval_splits):
        sp_tr, sp_te, stim_tr, stim_te = train_test_split(all_spikes[i], stimulus,
                                                          test_size=xval_fraction,
                                                          split_pos=xval_fraction*xvi)

        res = glm.minimize_loglhd(np.zeros((2, st.filter_length)), 0,
                                  stim_tr,
                                  st.frame_duration,
                                  sp_tr,
                                  usegrad=True,
                                  method='BFGS')
        if not res['success']:
            print(i, 'did not complete successfully.')
    #    kall[i,  ...] = res['x'][:-1]
    #    muall[i] = res['x'][-1]
        kall[i, :, xvi, ...], muall[i, xvi] = glm.splitpars(res['x'])

        frs[i, :] = glm.glm_neuron(kall[i, :, xvi, ...],
                                   muall[i, xvi],
                                   st.frame_duration)(stim_te)
        cross_corrs[i, xvi] = np.corrcoef(sp_te, frs[i, :])[0, 1]
    #%%
    fig, axes = plt.subplots(2, 2)

    for j in range(2):
        (axk, axnlt) = axes[j, :]
        for xvi in range(xval_splits):
            axk.plot(t, kall[i, j, xvi, :],
                     color='k', alpha=.2)
        axk.plot(t, omb_stas[i, j, :], label='OMB STA',
                 color='r', alpha=.6, ls='dashed')
        if j == 0:
            axk.set_title('Linear filters')
            axnlt.set_title('Nonlinearity')

        axk.set_xlabel('Time before spike [ms]')
        axk.set_ylabel(f'{plotlabels[j]}')

        # Use the mean filter for calculating nonlinearities
        k_avg = np.nanmean(kall[i, j, :, :], axis=0)

        generator = np.convolve(k_avg, stimulus[j, :], mode='full')[:-st.filter_length+1]
        nonlinearity, bins = nlt.calc_nonlin(all_spikes[i, :], generator, nr_bins=40)
        axnlt.plot(bins, nonlinearity/st.frame_duration)
        axnlt.set_xlabel('Stimulus projection')
        axnlt.set_ylabel('Firing rate [sp/s]')

    avg_cross_corr = np.nanmean(cross_corrs[i])
    avg_mu = np.nanmean(muall[i])

    plt.tight_layout()
    plt.subplots_adjust(top=.80)
    fig.suptitle(f'{st.exp_foldername} \n {st.stimname} \n'
                 f'{st.clids[i]} {glmlabel} avgmu: {avg_mu:4.2f} '
                 f'corr: {avg_cross_corr:4.2f} nsp: {all_spikes[i, :].sum():5.0f}')
    plt.show()
    fig.savefig(os.path.join(savepath, f'{st.clids[i]}.svg'))
    plt.close()

#%%
elapsed = dt.datetime.now()-start
print(f'Took {elapsed.total_seconds()/60:4.2f} minutes')

keystosave = ['kall', 'muall', 'cross_corrs', 'frs', 'glmlabel']

datadict = {}

for key in keystosave:
    datadict.update({key: locals()[key]})
npzfpath = os.path.join(savepath, f'{st.stimnr}_{glmlabel}.npz')
np.savez(npzfpath, **datadict)

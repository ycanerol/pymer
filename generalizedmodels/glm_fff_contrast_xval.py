#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""
import os

import numpy as np
import matplotlib.pyplot as plt
import datetime as dt

import analysis_scripts as asc
import nonlinearity as nlt
import genlinmod as glm
from train_test_split import train_test_split

from stimulus import Stimulus

#exp, stim_nr = '20180710', 8
exp = 'Kuehn'

xval_splits = 10
xval_fraction = 1/xval_splits

fff_stimnr = asc.stimulisorter(exp)['fff'][0]

fff = Stimulus(exp, fff_stimnr)
fff.filter_length = 20

# Get rid of list of numpy arrays
fff_stas = np.array(fff.read_datafile()['stas'])

glmlabel = 'GLM_contrast_xval'

savepath = os.path.join(fff.stim_dir, glmlabel)
os.makedirs(savepath, exist_ok=True)

fff.get_frametimings()
all_spikes = fff.allspikes()

start = dt.datetime.now()

kall = np.zeros((fff.nclusters, xval_splits, fff.filter_length))
muall = np.zeros((fff.nclusters, xval_splits))

frs = np.zeros((all_spikes.shape[0], int(all_spikes.shape[-1]/xval_splits)))

cross_corrs = np.zeros((fff.nclusters, xval_splits))

t = np.linspace(0, fff.filter_length*fff.frame_duration*1000, fff.filter_length)

plotlabels = ['Contrast']

# Temporary hack until FFF gets its own class.
fff.readpars()
if fff.param_file['Nblinks'] == 1:
    t_fff = np.linspace(0, fff.filter_length*fff.frame_duration*1000, fff.filter_length*2)
else:
    t_fff = t

stimulus = glm.loadstim(exp, fff_stimnr)

for i, cluster in enumerate(fff.clusters):

    for xvi in range(xval_splits):
        sp_tr, sp_te, stim_tr, stim_te = train_test_split(all_spikes[i],
                                                          stimulus,
                                                          test_size=xval_fraction,
                                                          split_pos=xval_fraction*xvi)

        res = glm.minimize_loglhd(np.zeros(fff.filter_length), 0,
                                  stim_tr,
                                  fff.frame_duration,
                                  sp_tr,
                                  usegrad=True,
                                  method='BFGS')
        if not res['success']:
            print(i, 'did not complete successfully.')
        kall[i, xvi, :] = res['x'][:-1]
        muall[i, xvi] = res['x'][-1]

        frs[i, :] = glm.glm_fr(kall[i, xvi, :],
                               muall[i, xvi],
                               fff.frame_duration)(stim_te)
        cross_corrs[i, xvi] = np.corrcoef(sp_te, frs[i, :])[0, 1]
    #%%
    fig, (axk, axnlt) = plt.subplots(1, 2)
    for xvi in range(xval_splits):
        axk.plot(t, kall[i, xvi, :],
                 color='k', alpha=.2)
    axk.plot(t_fff, fff_stas[i, :], label='FFF STA',
             color='r', alpha=.6, ls='dashed')
    axk.set_xlabel('Time before spike [ms]')
    axk.set_ylabel(f'{plotlabels[0]}')

    axk.set_title('Linear filter')
    axnlt.set_title('Nonlinearity')

    # Use the mean filter for calculating nonlinearities
    k_avg = np.nanmean(kall[i, :, :], axis=0)

    generator = np.convolve(k_avg, stimulus, mode='full')[:-fff.filter_length+1]
    nonlinearity, bins = nlt.calc_nonlin(all_spikes[i, :], generator, nr_bins=40)
    axnlt.plot(bins, nonlinearity/fff.frame_duration)
    axnlt.set_xlabel('Stimulus projection')
    axnlt.set_ylabel('Firing rate [sp/s]')

    avg_cross_corr = np.nanmean(cross_corrs[i])
    avg_mu = np.nanmean(muall[i])

    plt.subplots_adjust(top=.8)
    fig.suptitle(f'{fff.exp_foldername} \n {fff.stimname} \n'
                 f'{fff.clids[i]} {glmlabel} avgmu: {avg_mu:4.2f} '
                 f'corr: {avg_cross_corr:4.2f} nsp: {all_spikes[i, :].sum():5.0f}')
    plt.show()
    fig.savefig(os.path.join(savepath, f'{fff.clids[i]}.svg'))
    plt.close()
#%%

elapsed = dt.datetime.now()-start
print(f'Took {elapsed.total_seconds()/60:4.2f} minutes')

keystosave = ['kall', 'muall', 'cross_corrs', 'frs', 'glmlabel']

datadict = {}

for key in keystosave:
    datadict.update({key: locals()[key]})
npzfpath = os.path.join(savepath, f'{fff.stimnr}_{glmlabel}.npz')
np.savez(npzfpath, **datadict)

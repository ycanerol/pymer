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
import genlinmod as glm

from omb import OMB
from stimulus import Stimulus
from OMBanalyzer import q_nlt_recovery

#exp, stim_nr  = '20180710', 8
exp, stim_nr  = 'Kuehn', 13

fff_stimnr = asc.stimulisorter(exp)['fff'][0]

st = OMB(exp, stim_nr)
fff = Stimulus(exp, fff_stimnr)

# Get rid of list of numpy arrays
fff_stas = np.array(fff.read_datafile()['stas'])

glmlabel = 'GLM_onlycontrast'

savepath = os.path.join(st.stim_dir, glmlabel)
os.makedirs(savepath, exist_ok=True)

texture_data = st.read_texture_analysis()

all_spikes = st.allspikes()

start = dt.datetime.now()

kall = np.zeros((st.nclusters, st.filter_length))
muall = np.zeros(st.nclusters)

frs = np.zeros(all_spikes.shape)

cross_corrs = np.zeros(st.nclusters)

t = np.linspace(0, st.filter_length*st.frame_duration*1000, st.filter_length)

# Temporary hack until FFF gets its own class.
fff.readpars()
if fff.param_file['Nblinks'] == 1:
    t_fff = np.linspace(0, st.filter_length*st.frame_duration*1000, st.filter_length*2)
else:
    t_fff = t

for i, cluster in enumerate(st.clusters):

    stimulus = st.contrast_signal_cell(i).squeeze()

    res = glm.minimize_loglhd(np.zeros(st.filter_length), 0,
                              stimulus,
                              st.frame_duration,
                              all_spikes[i, :],
                              usegrad=True,
                              method='BFGS')
    if not res['success']:
        print(i, 'did not complete successfully.')
    kall[i, :] = res['x'][:-1]
    muall[i] = res['x'][-1]

    frs[i, :] = glm.glm_fr(kall[i, :],
                           muall[i],
                           st.frame_duration)(stimulus)
    cross_corrs[i] = np.corrcoef(all_spikes[i, :], frs[i, :])[0, 1]
    #%%
    fig, (axk, axnlt) = plt.subplots(1, 2)

    axk.plot(t, kall[i, :], label='GLM filter')
    axk.plot(t_fff, fff_stas[i, :], label='FFF STA',
             color='k', alpha=.6, ls='dashed')
    axk.set_xlabel('Time before spike [ms]')
    axk.set_ylabel('Filter strength')

    generator = np.convolve(kall[i, :], stimulus, mode='full')[:-st.filter_length+1]
    bins, spikecount = q_nlt_recovery(all_spikes[i, :], generator, nr_bins=40)
    axnlt.plot(bins, spikecount/st.frame_duration)
    axnlt.set_xlabel('Stimulus projection')
    axnlt.set_ylabel('Firing rate [spikes/s]')

    plt.subplots_adjust(top=.8)
    fig.suptitle(f'{st.exp_foldername} \n {st.stimname} \n'
                 f'{st.clids[i]} {glmlabel} mu: {muall[i]:4.2f} '
                 f'corr: {cross_corrs[i]:4.2f} nsp: {all_spikes[i, :].sum():5.0f}')
    fig.savefig(os.path.join(savepath, f'{st.clids[i]}.svg'))
    plt.close()
#%%

elapsed = dt.datetime.now()-start
print(f'Took {elapsed.total_seconds()/60:4.2f} minutes')

keystosave = ['kall', 'muall', 'cross_corrs', 'frs', 'glmlabel']

datadict = {}

for key in keystosave:
    datadict.update({key:locals()[key]})
npzfpath = os.path.join(savepath, f'{st.stimnr}_{glmlabel}.npz')
np.savez(npzfpath, **datadict)

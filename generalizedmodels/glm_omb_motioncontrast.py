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

from omb import OMB
from stimulus import Stimulus

#exp, stim_nr  = '20180710', 8
exp, stim_nr  = 'Kuehn', 13

fff_stimnr = asc.stimulisorter(exp)['fff'][0]

st = OMB(exp, stim_nr)
fff = Stimulus(exp, fff_stimnr)

# Get rid of list of numpy arrays
fff_stas = np.array(fff.read_datafile()['stas'])

glmlabel = 'GLM_motioncontrast'

savepath = os.path.join(st.stim_dir, glmlabel)
os.makedirs(savepath, exist_ok=True)

omb_stas = np.array(st.read_datafile()['stas'])
texture_data = st.read_texture_analysis()

all_spikes = st.allspikes()

start = dt.datetime.now()

kall = np.zeros((st.nclusters, 3, st.filter_length))
muall = np.zeros(st.nclusters)

frs = np.zeros(all_spikes.shape)

cross_corrs = np.zeros(st.nclusters)

stimulus = np.zeros((3, st.ntotal))
stimulus[:2, :] = st.bgsteps

t = np.linspace(0, st.filter_length*st.frame_duration*1000, st.filter_length)

plotlabels = ['Motion X', 'Motion Y', 'Contrast']

# Temporary hack until FFF gets its own class.
fff.readpars()
if fff.param_file['Nblinks'] == 1:
    t_fff = np.linspace(0, st.filter_length*st.frame_duration*1000, st.filter_length*2)
else:
    t_fff = t

for i, cluster in enumerate(st.clusters):

    stimulus[-1, :] = st.contrast_signal_cell(i).squeeze()

    res = glm.minimize_loglhd(np.zeros((3, st.filter_length)), 0,
                              stimulus,
                              st.frame_duration,
                              all_spikes[i, :],
                              usegrad=True,
                              method='BFGS')
    if not res['success']:
        print(i, 'did not complete successfully.')
#    kall[i,  ...] = res['x'][:-1]
#    muall[i] = res['x'][-1]
    kall[i, ...], muall[i] = glm.splitpars(res['x'])

    frs[i, :] = glm.glm_neuron(kall[i, ...],
                               muall[i],
                               st.frame_duration)(stimulus)
    cross_corrs[i] = np.corrcoef(all_spikes[i, :], frs[i, :])[0, 1]
    #%%
    fig, axes = plt.subplots(3, 2)

    for j in range(3):
        (axk, axnlt) = axes[j, :]
        axk.plot(t, kall[i, j, :], label='GLM filter')
        if j <= 1:
            axk.plot(t, omb_stas[i, j, :], label='OMB STA',
                     color='k', alpha=.6, ls='dashed')
        else:
            axk.plot(t_fff, fff_stas[i, :], label='FFF STA',
                     color='k', alpha=.6, ls='dashed')
        if j == 0:
            axk.set_title('Linear filters')
            axnlt.set_title('Nonlinearity')
        axk.set_xlabel('Time before spike [ms]')
        axk.set_ylabel(f'{plotlabels[j]}')

        generator = np.convolve(kall[i, j, :], stimulus[j, :], mode='full')[:-st.filter_length+1]
        nonlinearity, bins = nlt.calc_nonlin(all_spikes[i, :], generator, nr_bins=40)
        axnlt.plot(bins, nonlinearity/st.frame_duration)
        axnlt.set_xlabel('Stimulus projection')
        axnlt.set_ylabel('Firing rate [sp/s]')

    plt.tight_layout()
    plt.subplots_adjust(top=.80)
    fig.suptitle(f'{st.exp_foldername} \n {st.stimname} \n'
                 f'{st.clids[i]} {glmlabel} mu: {muall[i]:4.2f} '
                 f'corr: {cross_corrs[i]:4.2f} nsp: {all_spikes[i, :].sum():5.0f}')
    plt.show()
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

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""

import os

import numpy as np
import matplotlib.pyplot as plt

from scipy import signal

import analysis_scripts as asc
import iofuncs as iof
import plotfuncs as plf

from omb import OMB

import gen_quad_model_multidimensional as gqm


#def smoothspikes(spikes, sigma=1):
#    return gaussian_filter1d(spikes, sigma)


def smoothspikes(spikes, nbins=21, sigma=1):
    kernel = signal.windows.gaussian(nbins, sigma)
    smoothed = np.convolve(spikes, kernel, mode='same')
    return smoothed


def performance(modelspikes, realspikes, sigma=1):
    modelsp_smooth = smoothspikes(modelspikes, sigma)
    realsp_smooth = smoothspikes(realspikes, sigma)
    perf = np.correlate(modelsp_smooth, realsp_smooth, mode='same')
    return modelsp_smooth, realsp_smooth, perf


exp, stim = '20180710', 8
#exp, stim = 'Kuehn', 13

st = OMB(exp, stim)

# Just motion
pars_m = np.load(os.path.join(st.exp_dir, 'data_analysis',
                              st.stimname, 'GQM_Md', f'{stim}_GQM_Md.npz'))
# Motion and contrast
pars_cm = np.load(os.path.join(st.exp_dir, 'data_analysis',
                               st.stimname, 'GQM_Md_contrast', f'{stim}_GQM_Md_contrast.npz'))

data_texture = np.load(st.stim_dir + f'/{st.stimnr}_texturesta_20000fr.npz')
coords = data_texture['texture_maxi']


stim_m = pars_m['stimulus']
stim_cm = pars_cm['stimulus']

cc_m = np.zeros((st.nclusters))
cc_cm = np.zeros((st.nclusters))

frs_m = np.zeros((st.nclusters, stim_m.shape[-1]))
frs_cm = np.zeros((st.nclusters, stim_cm.shape[-1]))
frs_real = np.zeros((st.nclusters, st.ntotal))

sigmas = [0.01]
#%%
for sigma in sigmas:
    for i in range(st.nclusters):
        Q_m = pars_m['Qall'][i, ...]
        k_m = pars_m['kall'][i, ...]
        mu_m = pars_m['muall'][i, ...]

        Q_cm = pars_cm['Qall'][i, ...]
        k_cm = pars_cm['kall'][i, ...]
        mu_cm = pars_cm['muall'][i, ...]

        stimdim = 2  # Terrible hack, beacuse of the way gqm is written
        neuron_m = gqm.gqm_neuron(k_m, Q_m, mu_m, st.frame_duration)
        stimdim = 3  # Terrible hack, beacuse of the way gqm is written
        neuron_cm = gqm.gqm_neuron(k_cm, Q_cm, mu_cm, st.frame_duration)

        # This was not changing on 2019-07-03, but still does not work
        stim_cm[-1, :] = st.generatecontrast(coords[i, :])[..., :stim_cm.shape[-1]]

        fr_m = neuron_m(stim_m)
        fr_cm = neuron_cm(stim_cm)

        frs_m[i, :] = fr_m
        frs_cm[i, :] = fr_cm

        spikes = st.binnedspiketimes(i)

        toplot = slice(0, None)
        spikes_cut = spikes[toplot]

        fr_real = smoothspikes(spikes_cut, sigma=sigma)
        frs_real[i, :] = fr_real
#
        plt.plot(fr_m[toplot], label='motion')
        plt.plot(fr_cm[toplot], label='contrast+motion')
        plt.plot(fr_real, label='bined spikes', lw=0.5)
        break
        cc_m[i] = np.corrcoef(fr_real[:fr_m.shape[0]], fr_m)[0, 1]
        cc_cm[i] = np.corrcoef(fr_real[:fr_cm.shape[0]], fr_cm)[0, 1]
    #    plt.show()

    plt.scatter(cc_m, cc_cm, label=f'{sigma}')

plt.gca().set_aspect('equal')
plt.plot([0, 1], [0, 1], 'k--', alpha=.4)
plt.xlabel('Correlation coefficient [motion]')
plt.ylabel('Correlation coefficient [contrast + motion]')
plt.axis([-.05, 1.05, -.05, 1.05])
plt.title(f'r value {st.exp_foldername}')
plt.legend(fontsize='x-small')
#%%
i = 6
plt.plot(frs_m[i, :], label='GQM motion')
plt.plot(frs_cm[i, :], label='GQM contrast+motion')
#plt.plot(glm_frs[i, :], label='GLM contrast')
plt.plot(frs_real[i, :], label='bined spikes', lw=0.5)
plt.legend()

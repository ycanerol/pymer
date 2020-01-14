#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fit two separate GLMs to motion in X and Y directions

superseded by new version that considers multiple inputs in a single model
"""
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt

import iofuncs as iof
import analysis_scripts as asc
import genlinmod as glm


def normalizestas(stas):
    stas = np.array(stas)
    b = np.abs(stas).max(axis=2)
    stas_normalized = stas / b.repeat(stas.shape[2]).reshape(stas.shape)
    return stas_normalized

#%%
exp_name = '20180710'
stim_nr = 8

data = iof.load(exp_name, stim_nr)
stimulus = glm.loadstim(exp_name, stim_nr)

cell_lim = slice(None)

clusters = data['clusters'][cell_lim]
stas = np.array(data['stas'])
#stas = glm.normalizestas(data['stas'][cell_lim])
#frame_dur = data['frame_duration']

predstas = stas.copy()
predmus = np.zeros((stas.shape[0], stas.shape[-1]))

parameters = asc.read_parameters(exp_name, stim_nr)

_, frametimes = asc.ft_nblinks(exp_name, stim_nr, parameters.get('Nblinks', 2))
frametimes = frametimes[:-1]
frame_dur = np.ediff1d(frametimes).mean()

stashape = stas[:, 0, :].shape
#%%
start = dt.datetime.now()
for i, cluster in enumerate(clusters):
    for j, direction in enumerate(['x', 'y']):
        spikes = asc.read_raster(exp_name, stim_nr, cluster[0], cluster[1])
        spikes = asc.binspikes(spikes, frametimes)

        res = glm.minimize_loglhd(stas[i, j, :], 0, stimulus[j, :],
                                  frame_dur, spikes,
                                  usegrad=True,
                                  method='BFGS')
        k_pred = res['x'][:-1]
        mu_pred = res['x'][-1]

        predstas[i, j, :] = k_pred
        predmus[i, j] = mu_pred

#stas = normalizestas(stas)
#predstas = normalizestas(predstas)

elapsed = dt.datetime.now()-start
print(f'Took {elapsed.total_seconds()/60:4.2f} minutes')
#%%
lim = 50
for j, direction in enumerate(['X', 'Y']):
    imshowarg = {'cmap':'Greys_r'}
    ax_stas = plt.subplot(1, 3, 1)
    ax_stas.set_ylabel('Cells')
    ax_stas.set_xlabel('Time [ms]')
    ax_stas.set_xticklabels(['0', '0', '300', '450'])
    ax_stas.imshow(stas[:50, j, :],**imshowarg)
    ax_stas.set_title('STAs')
    ax_pred = plt.subplot(1, 3, 2)
    ax_pred.imshow(predstas[:50, j, :], **imshowarg)
    ax_pred.set_title('Predicted with\n log-likelihood maximization')
    ax_diff = plt.subplot(1, 3, 3)
    ax_diff.imshow((stas-predstas)[:50, j, :], vmin=-1, vmax=1, **imshowarg)
    ax_diff.set_title('Difference')
    ax_diff.set_xticklabels([''])
    ax_diff.set_yticklabels([''])
    ax_pred.set_yticklabels([''])
    ax_pred.set_xticklabels([''])
#    plt.suptitle(f'{exp_name}\n{iof.getstimname(exp_name, stim_nr)}')
    plt.subplots_adjust(top=.85)
#    plt.savefig('/media/owncloud/20181105_meeting_files/likelihood_OMB.pdf', bbox_inches='tight')
    plt.show()

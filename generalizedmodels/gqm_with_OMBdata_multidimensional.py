#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('/home/ycan/Documents/scripts/generalizedmodels/')

import gen_quad_model_multidimensional as gqm
import genlinmod as glm
import analysis_scripts as asc
import iofuncs as iof
import plotfuncs as plf
from scipy import linalg

from OMBanalyzer import q_nlt_recovery

exp_name = '20180710'
stim_nr = 8

exp_dir = iof.exp_dir_fixer(exp_name)
data = iof.load(exp_name, stim_nr)
stimname = iof.getstimname(exp_name, stim_nr)
stimulus_xy = glm.loadstim(exp_name, stim_nr)

gqmlabel= 'GQM_Md'
stimulus = stimulus_xy
stimdim = stimulus.shape[0]

clusters = data['clusters']

parameters = asc.read_parameters(exp_name, stim_nr)
_, frametimes = asc.ft_nblinks(exp_name, stim_nr,
                               parameters.get('Nblinks', 2))
frametimes = frametimes[:-1]
bin_length = np.ediff1d(frametimes).mean()

filter_length = l = data['filter_length']
refresh_rate = asc.read_spikesheet(exp_name)[1]['refresh_rate']

t = np.arange(0, filter_length*bin_length, bin_length)*1000

exp_name = os.path.split(exp_dir)[-1]
savedir = os.path.join(exp_dir, 'data_analysis', stimname, gqmlabel)
os.makedirs(savedir, exist_ok=True)

kall = np.zeros((clusters.shape[0], stimdim, l))
Qall = np.zeros((clusters.shape[0], stimdim, l, l))
muall = np.zeros((clusters.shape[0]))

eigvals = np.zeros((clusters.shape[0], stimdim, l))
eigvecs = np.zeros((clusters.shape[0], stimdim, l, l))


clids = plf.clusters_to_ids(clusters)

for i, cl in enumerate(clusters):
    sta = data['stas'][i]
    rawspikes = asc.read_raster(exp_name, stim_nr, *clusters[i][:2])

    spikes = asc.binspikes(rawspikes, frametimes)

    usegrad = True
    method = 'Newton-CG'

    import time
    start = time.time()
    res = gqm.minimize_loglikelihood(np.zeros((stimdim, l)), np.zeros((stimdim, l, l)), 0,
                                     stimulus, bin_length, spikes,
                                     usegrad=usegrad,
                                     minimize_disp=True, method=method)
    elapsed = time.time()-start

    print(f'Time elapsed: {elapsed/60:6.1f} mins')
    k_out, Q_out, mu_out = gqm.splitpars(res.x)
    kall[i, ...] = k_out
    Qall[i, ...] = Q_out
    muall[i] = mu_out
    #%%
    fig, axes = plt.subplots(stimdim, 5, figsize=(15,5))
    plt.rc('font', size=8)
    for j in range(stimdim):
        axk = axes[j, 0]
        axk.plot(t, sta[j], color='grey', label='STA')
        axk.plot(t, k_out[j, ...], label='k (GQM)')
        axk.legend(fontsize='x-small')
        axk.set_title('Linear motion filter')
        axk.set_xlabel('Time before spike [ms]')

        axk.text(0.8, 0.5, f'mu_out: {mu_out:4.2f}',
                 transform=axk.transAxes)

        axq = axes[j, 1]
        im = axq.imshow(Q_out[j, ...])
        plt.colorbar(im, ax=axq)
        axq.set_title('Quadratic motion filter (Q)')

        w_out, v_out = linalg.eigh(Q_out[j, ...])

        eigvals[i, j, :] = w_out
        eigvecs[i, j, :, :] = v_out

        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

        axv = axes[j, 2]
        axw = axes[j, 3]
        axn = axes[j, 4]

        axv.set_title('Eigenvalues of Q')
        axw.set_title('Eigenvectors of Q')
        axw.set_xlabel('Time before spike [ms]')

        axv.plot(w_out, 'ko')
        eiginds = [0, 1, l-2, l-1]

        for ind, (eigind, w) in enumerate(zip(eiginds, w_out[eiginds])):
            axv.plot(eigind, w, 'o', color=colors[ind])
            axw.plot(t, v_out[:, eigind], color=colors[ind])

            generator = np.convolve(eigvecs[i, j, :, eigind], stimulus[j, :],
                                    mode='full')[:-filter_length+1]

            bins, spikecount = q_nlt_recovery(spikes, generator, nr_bins=40)
            axn.plot(bins, spikecount/bin_length, color=colors[ind])

        # In order to set the legend the same for all components, we supply a
        # list to legend with a single element.

        axn.set_title('Nonlinearities')
        axn.set_ylabel('Firing rate [Hz]')
        axn.set_xlabel('Stimulus projection')

    plt.suptitle(f'{exp_name}\n'
              f'{stimname}\n'
              f'{clids[i]} {gqmlabel}')
    plt.tight_layout()
    plt.subplots_adjust(top=.85)
    #%%
    plt.savefig(os.path.join(savedir, clids[i]+'.svg'),
                bbox_inches = 'tight',
                pad_inches = 0.3)
    plt.show()
#    break
    #%%

keystosave = ['stimulus', 'Qall', 'kall', 'muall', 'eigvecs', 'eigvals',
              'bin_length', 'gqmlabel']

datadict = {}

for key in keystosave:
    datadict[key] = locals()[key]

npzfpath = os.path.join(savedir, str(stim_nr)+'_'+gqmlabel+'_data')
np.savez(npzfpath, **datadict)

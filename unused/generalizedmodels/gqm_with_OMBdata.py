#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""

import os
import numpy as np
import matplotlib.pyplot as plt

import gen_quad_model as gqm
import genlinmod as glm
import analysis_scripts as asc
import iofuncs as iof
import plotfuncs as plf
from scipy import linalg

import nonlinearity as nlt

exp_name = '20180710'
stim_nr = 8

exp_dir = iof.exp_dir_fixer(exp_name)
data = iof.load(exp_name, stim_nr)
stimname = iof.getstimname(exp_name, stim_nr)
stimulus_xy = glm.loadstim(exp_name, stim_nr)

gqmlabel= 'GQM_x'
stimulus = stimulus_xy[0, :]
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

kall = np.zeros((clusters.shape[0], l))
Qall = np.zeros((clusters.shape[0], l, l))
muall = np.zeros((clusters.shape[0]))

eigvals = np.zeros((clusters.shape[0], l))
eigvecs = np.zeros((clusters.shape[0], l, l))


clids = plf.clusters_to_ids(clusters)

for i, cl in enumerate(clusters):
    sta = data['stas'][i][0]
    rawspikes = asc.read_raster(exp_name, stim_nr, *clusters[i][:2])

    spikes = asc.binspikes(rawspikes, frametimes)

    usegrad = True
    method = 'Newton-CG'

    import time
    start = time.time()
    res = gqm.minimize_loglikelihood(np.zeros(l), np.zeros((l, l)), 0,
                                     stimulus, bin_length, spikes,
                                     usegrad=usegrad,
                                     minimize_disp=True, method=method)
    elapsed = time.time()-start

    print(f'Time elapsed: {elapsed/60:6.1f} mins')
    k_out, Q_out, mu_out = gqm.splitpars(res.x)
    kall[i, :] = k_out
    Qall[i, :, :] = Q_out
    muall[i] = mu_out
    #%%
    plt.figure(figsize=(9,7))
    axk = plt.subplot(321)
    axk.plot(t, sta, color='grey', label='STA')
    axk.plot(t, k_out, label='k (GQM)')
    axk.legend(fontsize='x-small')
    axk.set_title('Linear motion filter')
    axk.set_xlabel('Time before spike [ms]')

    axk.text(0.8, 0.5, f'mu_out: {mu_out:4.2f}',
             transform=axk.transAxes)

    axq = plt.subplot(322)
    im = axq.imshow(Q_out)
    plt.colorbar(im)
    axq.set_title('Quadratic motion filter (Q)')

    w_out, v_out = linalg.eigh(Q_out)

    eigvals[i, :] = w_out
    eigvecs[i, :, :] = v_out

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    axv = plt.subplot(323)
    axw = plt.subplot(324)
    axn = plt.subplot(326)

    axv.set_title('Eigenvalues of Q')
    axw.set_title('Eigenvectors of Q')
    axw.set_xlabel('Time before spike [ms]')

    axv.plot(w_out, 'ko')
    eiginds = [0, 1, l-2, l-1]
#    axw.plot(t, data['eigvecs_x'][i][:, [-1, 0, 1]], '--', color='grey')
    for ind, (eigind, w) in enumerate(zip(eiginds, w_out[eiginds])):
        axv.plot(eigind, w, 'o', color=colors[ind])
        axw.plot(t, v_out[:, eigind], color=colors[ind])

        generator = np.convolve(eigvecs[i, :, eigind], stimulus,
                                mode='full')[:-filter_length+1]

        nonlinearity, bins = nlt.calc_nonlin(spikes, generator, nr_bins=40)
        axn.plot(bins, nonlinearity/bin_length, color=colors[ind])

    # In order to set the legend the same for all components, we supply a
    # list to legend with a single element.
#    axw.legend(['STC'], fontsize='x-small')

    axn.set_title('Nonlinearities')
    axn.set_ylabel('Firing rate [Hz]')
    axn.set_xlabel('Stimulus projection')

    plt.suptitle(f'{exp_name}\n'
              f'{stimname}\n'
              f'{clids[i]} {gqmlabel}')
    plt.tight_layout()
    plt.subplots_adjust(top=.85)
    #%%
#    break
    plt.savefig(os.path.join(savedir, clids[i]+'.pdf'),
                bbox_inches = 'tight',
                pad_inches = 0.3)
    plt.show()
    #%%

keystosave = ['stimulus', 'Qall', 'kall', 'muall', 'eigvecs', 'eigvals',
              'bin_length', 'gqmlabel']

datadict = {}

for key in keystosave:
    datadict[key] = locals()[key]

npzfpath = os.path.join(savedir, str(stim_nr)+'_'+gqmlabel)
np.savez(npzfpath, **datadict)

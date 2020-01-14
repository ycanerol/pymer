#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Use movement in both x and y directions for OMB analysis
"""

import numpy as np
import matplotlib.pyplot as plt

import gen_quad_model_multidimensional as gqm
import genlinmod as glm
import analysis_scripts as asc
import iofuncs as iof
import plotfuncs as plf
from scipy import linalg


exp_name = '20180710'
stim_nr = 8
data = iof.load(exp_name, stim_nr)
stimulus_xy = glm.loadstim(exp_name, stim_nr)
stimulus = stimulus_xy
clusters = data['clusters']

parameters = asc.read_parameters(exp_name, stim_nr)
_, frametimes = asc.ft_nblinks(exp_name, stim_nr, parameters.get('Nblinks', 2))
frametimes = frametimes[:-1]
bin_length = np.ediff1d(frametimes).mean()

filter_length = l = data['filter_length']
refresh_rate = asc.read_spikesheet(exp_name)[1]['refresh_rate']
exp_name = iof.exp_dir_fixer(exp_name).split('/')[-1]

# Limit to the first cell for now
#clusters = clusters[[0, 1],  ...]

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


    #%%
    axk = plt.subplot(321)
    axk.plot(sta, label='STA')
    axk.plot(k_out, label='k_out')
    axk.legend(fontsize='x-small')

    axk.text(0.8, 0.5, f'mu_out: {mu_out:4.2f}', transform=axk.transAxes)

    axq = plt.subplot(322)
    axq.imshow(Q_out)

    #plt.show()

    w_out, v_out = linalg.eigh(Q_out)

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color'][2:]

    axv = plt.subplot(323)
    axw = plt.subplot(324)

    axv.plot(w_out, 'ko')
    eiginds = [0, 1, l-1]
    for ind, (eigind, w) in enumerate(zip(eiginds, w_out[eiginds])):
        axv.plot(eigind, w, 'o', color=colors[ind])
        axw.plot(v_out[:, eigind], lw=.8, color=colors[ind])
    axw.plot(data['eigvecs_x'][i][:, [-1, 0, 1]], '--', label='STC0')
    axw.legend(fontsize='x-small')
    plt.suptitle(f'{exp_name}\n'
              f'{iof.getstimname(exp_name, stim_nr)}\n'
              f'{plf.clusters_to_ids(clusters)[i]} GQM')
    plt.tight_layout()
    plt.subplots_adjust(top=.85)

    #plt.savefig(savepath+'gqm_omb.pdf', bbox_inches = 'tight', pad_inches = 0.3)
    #plt.savefig(savepath+'gqm_omb.png', bbox_inches = 'tight', pad_inches = 0.3)
    plt.show()

#    plt.plot(v_out[:, [0, -1]])
#    plt.show()

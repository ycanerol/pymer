#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""

import os
import numpy as np
import matplotlib.pyplot as plt

import gen_quad_model_multidimensional as gqm
import analysis_scripts as asc
import iofuncs as iof
import plotfuncs as plf
from scipy import linalg
from train_test_split import train_test_split

import nonlinearity as nlt

from omb import OMB


exp_name, stim_nr = '20180710', 8
#exp_name, stim_nr = 'Kuehn', 13

val_split_size = 0.1
val_split_pos = 0.5


st = OMB(exp_name, stim_nr, maxframes=None)

data = st.read_datafile()

stimdim = 3

# Add placeholder contrast row, this will be different for
# each cell
stimulus = np.zeros((stimdim, st.ntotal))
stimulus[:2, ...] = st.bgsteps

gqmlabel = 'GQM_motioncontrast_val'

fl = st.filter_length

t = np.arange(0, st.filter_length*st.frame_duration, st.frame_duration)*1000

savedir = os.path.join(st.exp_dir, 'data_analysis', st.stimname, gqmlabel)
os.makedirs(savedir, exist_ok=True)

kall = np.zeros((st.nclusters, stimdim, fl))
Qall = np.zeros((st.nclusters, stimdim, fl, fl))
muall = np.zeros((st.nclusters))

eigvals = np.zeros((st.nclusters, stimdim, fl))
eigvecs = np.zeros((st.nclusters, stimdim, fl, fl))

cross_corrs = np.zeros(st.nclusters)

for i, cl in enumerate(st.clusters):
    sta = data['stas'][i]

    spikes = st.binnedspiketimes(i)

    import time
    start = time.time()

    # Calculate the contrast for each cell's receptive field
    stimulus[-1, :] = st.contrast_signal_cell(i)

    sp_tr, sp_te, stim_tr, stim_te = train_test_split(spikes, stimulus,
                                                      test_size=val_split_size,
                                                      split_pos=val_split_pos)

    res = gqm.minimize_loglikelihood(np.zeros((stimdim, fl)),
                                     np.zeros((stimdim, fl, fl)), 0,
                                     stim_tr,
                                     st.frame_duration,
                                     sp_tr,
                                     minimize_disp=True,
                                     method='BFGS')
    elapsed = time.time()-start

    print(f'Time elapsed: {elapsed/60:6.1f} mins for cell {i}')
    k_out, Q_out, mu_out = gqm.splitpars(res.x)
    kall[i, ...] = k_out
    Qall[i, ...] = Q_out
    muall[i] = mu_out

    firing_rate = gqm.gqm_neuron(k_out, Q_out, mu_out, st.frame_duration)(stim_te)
    cross_corr = np.corrcoef(sp_te, firing_rate)[0, 1]
    cross_corrs[i] = cross_corr
    #%%
    fig, axes = plt.subplots(stimdim, 5, figsize=(15, 5))
    plt.rc('font', size=8)
    for j in range(stimdim):
        axk = axes[j, 0]
        if j <= 1:
            axk.plot(t, sta[j], color='grey', label='STA')
        axk.plot(t, k_out[j, ...], label='k (GQM)')
        axk.legend(fontsize='x-small')
        axk.set_title('Linear filter')
        axk.set_xlabel('Time before spike [ms]')

        axk.text(0.8, 0.5, f'mu_out: {mu_out:4.2f}',
                 transform=axk.transAxes)

        axq = axes[j, 1]
        im = axq.imshow(Q_out[j, ...])
        plt.colorbar(im, ax=axq)
        axq.set_title('Quadratic filter (Q)')

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
        eiginds = [0, 1, fl-2, fl-1]

        for ind, (eigind, w) in enumerate(zip(eiginds, w_out[eiginds])):
            axv.plot(eigind, w, 'o', color=colors[ind])
            axw.plot(t, v_out[:, eigind], color=colors[ind])

            generator = np.convolve(eigvecs[i, j, :, eigind], stimulus[j, :],
                                    mode='full')[:-st.filter_length+1]

            nonlinearity, bins = nlt.calc_nonlin(spikes, generator, nr_bins=40)
            axn.plot(bins, nonlinearity/st.frame_duration, color=colors[ind])

        axn.set_title('Nonlinearities')
        axn.set_ylabel('Firing rate [Hz]')
        axn.set_xlabel('Stimulus projection')

    plt.suptitle(f'{st.exp_foldername}\n'
                 f'{st.stimname}\n'
                 f'{st.clids[i]} {gqmlabel} corr: {cross_corr:4.2f} nsp: {spikes.sum():5.0f}')
    plt.tight_layout()
    plt.subplots_adjust(top=.85)
    #%%
    plt.savefig(os.path.join(savedir, st.clids[i]+'.svg'),
                bbox_inches='tight',
                pad_inches=0.3)
    plt.show()
#    break
    #%%

keystosave = ['Qall', 'kall', 'muall', 'eigvecs', 'eigvals',
              'gqmlabel', "cross_corrs", 'val_split_size', 'val_split_pos']

datadict = {}

for key in keystosave:
    datadict[key] = locals()[key]

npzfpath = os.path.join(savedir, str(st.stimnr)+'_'+gqmlabel)
np.savez(npzfpath, **datadict)

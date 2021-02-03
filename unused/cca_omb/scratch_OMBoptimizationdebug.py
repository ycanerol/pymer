#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Inspect the optimization process for maximum likelihood estimation in
GQM fitting.

Adding the contrast signal deteriorates the model performance, is this
a optimization bug?

Resolved: problem was not optimization, the low performance was due to
          mistake in contrast signal generation.

"""

import sys
import time


import numpy as np
import matplotlib.pyplot as plt



import gen_quad_model_multidimensional as gqm
import analysis_scripts as asc
import iofuncs as iof
import plotfuncs as plf
from scipy import linalg

from omb import OMB


def optim_tracker(pars):
    global pars_progress
    pars_progress.append(pars)


def performance(spikes, pars, stimulus):
    """
    Calculate the response of a model neuron, given a set of filters and
    a stimulus. Also calculate the cross-correlation with the real spikes
    as a measure of performance.

    """
    k, Q, mu = gqm.splitpars(pars)
    firing_rate = gqm.gqm_neuron(k, Q, mu, st.frame_duration)(stimulus)
    cross_corr = np.corrcoef(spikes, firing_rate)[0, 1]
    return cross_corr, firing_rate


exp_name, stim_nr = 'Kuehn', 13

st = OMB(exp_name, stim_nr, maxframes=None)

data = iof.load(exp_name, stim_nr)

data_texture = np.load(st.stim_dir + f'/{st.stimnr}_texturesta_20000fr.npz')
coords = data_texture['texture_maxi']

stimulus = np.vstack((st.bgsteps,
                      np.zeros(st.ntotal)))

stimdim = stimulus.shape[0]

kall = np.zeros((st.nclusters, stimdim, st.filter_length))
Qall = np.zeros((st.nclusters, stimdim, st.filter_length, st.filter_length))
muall = np.zeros((st.nclusters))

all_pars_progress = []
all_res = []

cells_of_interest = [0, 1, 3]

for i, cl in enumerate(st.clusters):
    if i not in cells_of_interest:
        continue

    sta = data['stas'][i]
    spikes = st.binnedspiketimes(i)

    stimulus[-1, :] = st.generatecontrast(coords[i, :])

    # Track the progress of the parameters, optim_tracker uses global variable
    pars_progress = []

    start = time.time()

    res = gqm.minimize_loglikelihood(np.zeros((stimdim, st.filter_length)),
                                     np.zeros((stimdim, st.filter_length,
                                                        st.filter_length)),
                                     0,
                                     stimulus,
                                     st.frame_duration,
                                     spikes,
                                     method='BFGS',
                                     callback=optim_tracker)

    elapsed = time.time()-start
    print(f'Time elapsed: {elapsed/60:6.1f} mins for cell {i}')

    all_pars_progress.append(pars_progress)
    all_res.append(res)

    plt.plot(gqm.splitpars(res.x)[0].T)
    plt.title('Final linear filters')
    plt.show()

    spikes = st.binnedspiketimes(i)

    if res.nit > 1000:
        break
    cc_progress = np.zeros(res.nit)

    for j, pars in enumerate(pars_progress):
        cc_progress[j], fr = performance(spikes, pars, stimulus)

#        plt.plot(ki.T)
#        plt.title(j)
#        plt.show()
    plt.plot(cc_progress)
    plt.title(f'cell {i}')
    plt.show()

#%%
ccs = np.zeros(st.nclusters)
for i, cl in enumerate(st.clusters):
    pars = all_res[i].x
    spikes = st.binnedspiketimes(i)
    stimulus[-1, :] = st.generatecontrast(coords[i, :])

    ccs[i], _ = performance(spikes, pars, stimulus)

plt.hist(ccs)
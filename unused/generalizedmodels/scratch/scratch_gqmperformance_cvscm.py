#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare the performance of GQM with different inputs, to see
how including contrast or motion influences the prediction performance.

"""
import os

import numpy as np
import matplotlib.pyplot as plt

from omb import OMB
from driftinggratings import DriftingGratings

import gen_quad_model_multidimensional as gqm

from scipy import stats

#exp, stim = '20180710', 8
exp, stim = 'Kuehn', 13

st = OMB(exp, stim)
species = st.metadata["animal"]


gqmlabels = ['GQM_contrast_val', 'GQM_motion_val', 'GQM_motioncontrast_val']

# Only contrast
data_c = np.load(os.path.join(st.exp_dir, 'data_analysis',
                              st.stimname, gqmlabels[0], f'{stim}_{gqmlabels[0]}.npz'))
# only Motion
data_m = np.load(os.path.join(st.exp_dir, 'data_analysis',
                              st.stimname, gqmlabels[1], f'{stim}_{gqmlabels[1]}.npz'))
# Motion and contrast
data_cm = np.load(os.path.join(st.exp_dir, 'data_analysis',
                               st.stimname, gqmlabels[2], f'{stim}_{gqmlabels[2]}.npz'))

# Exclude those with very few spikes
cutoff = 0.1  # In units of spikes/s
lowq = (st.allspikes().mean(axis=1) / st.frame_duration) < cutoff

cc_c = data_c['cross_corrs'][~lowq]
cc_m = data_m['cross_corrs'][~lowq]
cc_cm = data_cm['cross_corrs'][~lowq]

#%% Scatter
fig, axes = plt.subplots(2, 2,
                         figsize=(5.5, 5),
#                         sharex=True, sharey=True
                         )

ax1, ax2, ax3, ax4 = axes.flat

unityline = [0.05, .5]
lims = [-0.05, .55]
ticks = [0, .25, .5]
for ax in (ax1, ax3, ax4):
    ax.axis('equal')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.plot(unityline, unityline, 'k', alpha=.3, ls='dashed')

scatterkwargs = dict(c='k', s=10)

ax1.scatter(cc_c, cc_m, **scatterkwargs)
ax2.set_axis_off()
ax3.scatter(cc_c, cc_cm, **scatterkwargs)
ax4.scatter(cc_m, cc_cm, **scatterkwargs)

# If salamander, highlight DS cells
if species == 'salamander':

    dg = DriftingGratings(exp, 5)
    dsc_i = np.where(dg.dsi[~lowq] > .3)[0]
    scatterkwargs.update({'c' : 'red'})

    ax1.scatter(cc_c[dsc_i], cc_m[dsc_i], **scatterkwargs)
    ax3.scatter(cc_c[dsc_i], cc_cm[dsc_i], **scatterkwargs)
    ax4.scatter(cc_m[dsc_i], cc_cm[dsc_i], **scatterkwargs)


ax1.set_ylabel('Motion')
ax3.set_ylabel('Contrast and motion')
ax3.set_xlabel('Contrast')
ax4.set_xlabel('Motion')

fig.suptitle(f'GQM performance for {species} cells')
fig.savefig(f'/media/owncloud/20191021_labmeeting/gqm_performance_{species}.pdf')

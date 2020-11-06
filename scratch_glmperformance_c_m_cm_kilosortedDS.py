#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""
import os
import numpy as np

import matplotlib.pyplot as plt

import iofuncs as iof
from omb import OMB
# from driftinggratings import DriftingGratings

exp, ombstimnr  = '20180710_kilosorted', 8
save = True
savedir = '/home/ycan/Downloads/2020-05-25_labmeeting/'

st = OMB(exp, ombstimnr)
# dg = DriftingGratings(exp, dgstimnr)

# mat = iof.readmat(f'{st.exp_dir}/CellStats_RF-SVD_DS-CircVar.mat')
# dsc_i = mat['DScells'] - 1 # Convert matlab indexing to Python
# dsc_i_dg = np.where(dg.dsi>.3)[0]
dsc_i = [8, 33, 61, 73, 79]

#dsc_i = dsc_i_dg # HINT

# clusters = np.loadtxt(f'{st.exp_dir}/goodChannels.txt')

data_m = np.load(os.path.join(st.exp_dir, 'data_analysis',
                              st.stimname, 'GLM_motion_xval', f'{ombstimnr}_GLM_motion_xval.npz'))

# Only contrast
data_c = np.load(os.path.join(st.exp_dir, 'data_analysis',
                              st.stimname, 'GLM_contrast_xval', f'{ombstimnr}_GLM_contrast_xval.npz'))
# Motion and contrast
data_cm = np.load(os.path.join(st.exp_dir, 'data_analysis',
                               st.stimname, 'GLM_motioncontrast_xval', f'{ombstimnr}_GLM_motioncontrast_xval.npz'))

cc_c = data_c['cross_corrs'].mean(axis=1)
cc_m = data_m['cross_corrs'].mean(axis=1)
cc_cm = data_cm['cross_corrs'].mean(axis=1)

#%%
fig1, axes1 = plt.subplots(2, 2,
#                          sharex=True, sharey=True
                          )

scatterkwargs = dict(s=5)

ax1, ax2, ax3, ax4 = axes1.flat
ax2.axis('off')
ax1.scatter(cc_c, cc_m, c='k', alpha=0.2, **scatterkwargs)
ax1.scatter(cc_c[dsc_i], cc_m[dsc_i], c='red', **scatterkwargs)
ax1.set_ylabel('Motion')

ax3.scatter(cc_c, cc_cm, c='k', alpha=0.2, **scatterkwargs)
ax3.scatter(cc_c[dsc_i], cc_cm[dsc_i], c='red', **scatterkwargs)
ax3.set_ylabel('Contrast and motion')
ax3.set_xlabel('Contrast')

ax4.scatter(cc_m, cc_cm, c='k', alpha=0.2, **scatterkwargs)
ax4.scatter(cc_m[dsc_i], cc_cm[dsc_i], c='red', **scatterkwargs)
ax4.set_xlabel('Motion')

for ax in [ax1, ax3, ax4]:
    ax.axis('equal')
    ax.plot([0, .4], [0, .4], 'k', alpha=.2, ls='dashed')

fig1.suptitle('GLM performance for marmoset cells')
fig1.show()
if save:
    fig1.savefig(savedir + 'glm_performance.pdf')
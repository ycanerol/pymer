#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""
import os

import numpy as np
import matplotlib.pyplot as plt

import iofuncs as iof
import plotfuncs as plf
from omb import OMB

save = False
savedir = '/home/ycan/Downloads/'

labels = ['GLM_contrast_xval', 'GLM_motion_xval', 'GLM_motioncontrast_xval']

exp, stim_nr = 'Kuehn', 13
#exp, stim_nr = '20180710', 8
st = OMB(exp, stim_nr)

matfile = iof.readmat(st.exp_dir+'/CellStats_RF-SVD_DS-CircVar.mat')
dscells = matfile['DScells'] - 1 # Matlab to Python indexes

all_cross_corrs = []

for label in labels:
    data = np.load(os.path.join(st.stim_dir, label, f'{st.stimnr}_{label}.npz'))
    cross_corr = data['cross_corrs'].mean(axis=1)
    # Filter DS cells
#    cross_corr = cross_corr[dscells]
    all_cross_corrs.append(cross_corr)

cc_c, cc_m, cc_cm = all_cross_corrs

#%%
fig1, axes1 = plt.subplots(2, 2,
#                          sharex=True, sharey=True
                          )

scatterkwargs = dict(s=10, c='k')

ax1, ax2, ax3, ax4 = axes1.flat
ax2.axis('off')
ax1.scatter(cc_c, cc_m, **scatterkwargs)
ax1.set_ylabel('Motion')

ax3.scatter(cc_c, cc_cm, **scatterkwargs)
ax3.set_ylabel('Contrast and motion')
ax3.set_xlabel('Contrast')

ax4.scatter(cc_m, cc_cm, **scatterkwargs)
ax4.set_xlabel('Motion')

for ax in [ax1, ax3, ax4]:
    ax.axis('equal')
    ax.plot([0, .3], [0, .3], 'k', alpha=.3, ls='dashed')

fig1.suptitle('Cross validated GLM performance dependence on inputs')
fig1.show()
if save:
    fig1.savefig(savedir + 'comparison_scatter_salamander.pdf')
#%%
fig2, axes2 = plt.subplots(3, 1, sharex=True, sharey=True)
ax1, ax2, ax3 = axes2
ax1.hist(cc_c)
ax1.set_ylabel('Contrast')
ax1.set_title('Distribution of Pearson\'s R')
ax2.hist(cc_m)
ax2.set_ylabel('Motion')
ax3.hist(cc_cm)
ax3.set_ylabel('Contrast and motion')
fig2.show()

if save:
    fig2.savefig(savedir + 'comparison_hist_salamander.pdf')

#%%
fig3 = plt.figure(figsize=(6, 5))
for i in range(cc_c.shape[0]):
    plt.plot([0, 1, 2], [cc_c[i], cc_m[i], cc_cm[i]], color='k', lw=.2)
ax = plt.gca()
ax.set_xlim([-.5, 2.5])
ax.set_xticks([0, 1, 2])
ax.set_xticklabels(['Contrast', 'Motion', 'Motion and contrast'])
ax.set_ylabel('Performance')
ax.set_title(f'Model prediction performance ({st.exp_foldername})')
fig3.show()
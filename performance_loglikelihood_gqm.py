#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Measuring model performance

cross-validated log likelihood
Butts et al. 2011 J.Neurosci


"""
import numpy as np
import matplotlib.pyplot as plt
from omb import OMB
import gen_quad_model_multidimensional as gqm
import plotfuncs as plf
#from driftinggratings import DriftingGratings


exp, stim = '20180710', 8
#exp, stim = 'Kuehn', 13
st = OMB(exp, stim)
species = st.metadata["animal"]
allspikes = st.allspikes()

data_cm = np.load(f'{st.stim_dir}/GQM_motioncontrast_val/{st.stimnr}_GQM_motioncontrast_val.npz')
data_c = np.load(f'{st.stim_dir}/GQM_contrast_val/{st.stimnr}_GQM_contrast_val.npz')
data_m = np.load(f'{st.stim_dir}/GQM_motion_val/{st.stimnr}_GQM_motion_val.npz')

model_input = [('Contrast and motion', 3),
               ('Contrast', 1),
               ('Motion', 2)]

logls = np.zeros((st.nclusters, 3))

# Exclude those with very few spikes
cutoff = 0.2  # In units of spikes/s
avg_firingrates = (st.allspikes().mean(axis=1) / st.frame_duration)
lowq = avg_firingrates < cutoff
lowq_mask = np.broadcast_to(lowq[:, None], logls.shape)

for j, (label, stimdim) in enumerate(model_input):
    data = [data_cm, data_c, data_m][j]
    kall = data['kall']
    qall = data['Qall']
    muall = data['muall']

    gqm.set_stimdim(stimdim)
    gqm.set_filter_length(st.filter_length)

    stimulus = np.zeros((stimdim, st.ntotal))
    if 'motion' in label.lower():
        stimulus[:2, :] = st.bgsteps

    for i in range(st.nclusters):

        if 'contrast' in label.lower():
            stimulus[-1, :] = st.contrast_signal_cell(i)
        spikes = allspikes[i, :]
        logl = gqm.loglikelihood(gqm.flattenpars(kall[i], qall[i],
                                                 muall[i]),
                                stimulus, spikes, st.frame_duration)
        logls[i, j] = -logl  # negative loglikelihood is returned by the function

logls_norm = logls / allspikes.sum(axis=1)[:, None]  # Normalize with the number of spikes
logls_norm = np.ma.array(logls_norm, mask=lowq_mask)

#%%
plt.scatter(logls_norm[:, 0], logls_norm[:, 2])
plt.scatter(logls_norm[:, 1], logls_norm[:, 2])
plt.scatter(logls_norm[:, 0], logls_norm[:, 1])

#%%
fig, axes = plt.subplots(2, 2,
                         figsize=(5.5, 5),
#                         sharex=True, sharey=True
                         )

ax1, ax2, ax3, ax4 = axes.flat

unityline = [-1, 2.5]
lims = [-2.5, 2.5]
#    ticks = [-4, .25, .5]
for ax in (ax1, ax3, ax4):
    ax.axis('equal')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    plf.integerticks(ax, 4)
#    ax.set_xticks(ticks)
#    ax.set_yticks(ticks)
    ax.plot(unityline, unityline, 'k', alpha=.3, ls='dashed')

scatterkwargs = dict(c='k', s=10)

ax1.scatter(logls_norm[:, 1], logls_norm[:, 2], **scatterkwargs)
ax2.set_axis_off()
ax3.scatter(logls_norm[:, 1], logls_norm[:, 0], **scatterkwargs)
ax4.scatter(logls_norm[:, 2], logls_norm[:, 0], **scatterkwargs)

# If salamander, highlight DS cells
if species == 'salamander':
    import iofuncs as iof
    mat = iof.readmat(f'{st.exp_dir}/CellStats_RF-SVD_DS-CircVar.mat')
    dsc_i = mat['DScells'] - 1 # Convert matlab indexing to Python
    dsc_i = np.array([True if i in dsc_i else False for i in range(st.nclusters)])
#    dsc_i = dsc_i[~lowq]

    scatterkwargs.update({'c':'red'})
    logls_norm_ds = logls_norm.copy()
    logls_norm_ds[~dsc_i, :] = np.ma.masked

    ax1.scatter(logls_norm_ds[:, 1], logls_norm_ds[:, 2], **scatterkwargs)
    ax3.scatter(logls_norm_ds[:, 1], logls_norm_ds[:, 0], **scatterkwargs)
    ax4.scatter(logls_norm_ds[:, 2], logls_norm_ds[:, 0], **scatterkwargs)

ax1.set_ylabel('Motion')
ax3.set_ylabel('Contrast and motion')
ax3.set_xlabel('Contrast')
ax4.set_xlabel('Motion')

fig.suptitle(f'GQM Log likelihood per spike \n{species} {st.exp_foldername}')
fig.savefig(f'/home/ycan/Documents/meeting_notes/2019-11-13/loglikelihood_gqm_{species}.pdf')
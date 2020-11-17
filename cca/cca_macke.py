from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import rcca

from matplotlib.ticker import MaxNLocator

import analysis_scripts as asc
import model_fitting_tools as mft
import spikeshuffler
import nonlinearity as nlt
import plotfuncs as plf
from omb import OMB

filter_length = (20, 20)

st = OMB('20180710_kilosorted', 8, maxframes=None)

spikes = st.allspikes()
# Set the mean to zero for spikes
#spikes -= spikes.mean(axis=1)[:, None]

bgsteps = st.bgsteps


stimulus = mft.packdims(st.bgsteps, filter_length[0])
spikes = mft.packdims(spikes, filter_length[1])

stimavg = stimulus.mean(axis=0)[np.newaxis, :]
spikesavg = spikes.mean(axis=0)[np.newaxis, :]

#%%
stimcov = np.cov(stimulus.T)
spkcov = np.cov(spikes.T)

stspcov = np.zeros((stimcov.shape[0], spkcov.shape[0]))

for i in range(stspcov.shape[0]):
    for j in range(stspcov.shape[1]):
        stspcov[i, j] = np.mean(
                (stimulus[:, i] - np.mean(stimulus[:, i]))*
                (spikes[:, j] - np.mean(spikes[:, j]))
                                )
#%%
whitened_cov = stimcov**(-0.5) @ stspcov @ spkcov**(-0.5)

u, s, vh = np.linalg.svd(whitened_cov, full_matrices=True)

print('done')

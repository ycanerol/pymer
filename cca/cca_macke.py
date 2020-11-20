from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import rcca
from scipy.linalg import fractional_matrix_power

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

# Exclude rows that have no spikes throughout
nonzerospikerows = ~np.isclose(spikes.sum(axis=1), 0)
spikes = spikes[nonzerospikerows, :]

ncells = spikes.shape[0]

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
ncomponents = min(stimulus.shape[1], spikes.shape[1])

stimcov_exp = fractional_matrix_power(stimcov, -0.5)
spkcov_exp = fractional_matrix_power(spkcov, -0.5)
whitened_cov =  stimcov_exp @ stspcov @ spkcov_exp

u, s, v = np.linalg.svd(whitened_cov, full_matrices=True)

# rows of v correspond to components, to interpret it in the same way,
# we transpose it
v = v.T
print('done')

# SVD returns a much larger matrix than needed for the larger matrix, we take
# only the needed part.
respcomps = v[:, :ncomponents].reshape((ncells, filter_length[1], ncomponents))
# Reshape so that the order is: ncomponents, ncells, time
respcomps = np.moveaxis(respcomps, [2, 0, 1], [0, 1, 2])

stimcomps = u.reshape((2, filter_length[0], ncomponents))
stimcomps = np.moveaxis(stimcomps, [2, 0, 1], [0, 1, 2])
#%%

fig, axes = plt.subplots(3, 2)
ax0 = axes[0, 0]
ax1 = axes[0, 1]
ax2 = axes[1, 0]
ax3 = axes[1, 1]
ax4 = axes[2, 0]
ax0.imshow(u)
ax1.imshow(v)
ax2.plot(s, 'ok')
ax2.set_title('Singular values')

# ax3.imshow(v[:, 0].reshape((ncells, filter_length[1])))
ax3.imshow(respcomps[0, ...])
ax4.plot(stimcomps[0, ...].T)
plt.show()


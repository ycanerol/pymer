"""

"""
import numpy as np
from sklearn.cross_decomposition import CCA

from omb import OMB
import analysis_scripts as asc
import plotfuncs as plf
from model_fitting_tools import packdims 


exp, stim_nr = '20180710_kilosorted', 8
n_components = 6


st = OMB(exp, stim_nr)
filter_length = st.filter_length

spikes = st.allspikes()
bgsteps = st.bgsteps

# stimulus = asc.rolling_window(bgsteps, filter_length,
#                               preserve_dim=True)
stimulus = packdims(bgsteps, filter_length)


cca = CCA(n_components=n_components,
          scale=True, max_iter=500, tol=1e-06, copy=True)

cca.fit(spikes.T, stimulus)

x, y = cca.transform(spikes.T, stimulus)
x, y = x.T, y


#%%
import matplotlib.pyplot as plt

cells = cca.x_weights_.T

motionfilt_x = cca.y_weights_[:filter_length].T
motionfilt_y = cca.y_weights_[filter_length:].T

fig, axes = plt.subplots(*plf.numsubplots(n_components))

for i, ax in enumerate(axes.flat):
    ax.plot(motionfilt_x[i, :])
    ax.plot(motionfilt_y[i, :])
    ax.set_ylim(cca.y_weights_.min(), cca.y_weights_.max())
fig.suptitle('Stimulus components for each CCA filter')
plt.show()

im = plt.imshow(cells, cmap='RdBu_r', vmin=asc.absmin(cells), vmax=asc.absmax(cells))
plt.ylabel('Components')
plt.xlabel('Cells')
plf.colorbar(im)
plt.show()



plt.plot(cca.coef_[:, 0], cca.coef_[:, 1], 'ko')
plt.show()

offsets = np.arange(n_components)*.6
plt.plot(cca.y_weights_ + np.tile(offsets, (stimulus.shape[1], 1)), lw=0.5)
plt.hlines(offsets, 0, stimulus.shape[1], lw=.2)

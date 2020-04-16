"""

"""
import numpy as np
from sklearn.cross_decomposition import CCA

from omb import OMB
import analysis_scripts as asc
import plotfuncs as plf


def packdims(array, window):
    sh = array.shape
    if array.ndim == 1:
        array = array[None, :]
    if array.ndim > 2:
        array = array.reshape(np.prod(sh[:-1]), -1)
    rw = np.empty((sh[-1], array.shape[0]*window))
    for i in range(array.shape[0]):
        rw[:, i*window:(i+1)*window] = asc.rolling_window(array[i, :], window)
    return rw


exp, stim_nr = '20180710', 8
n_components = 6


st = OMB(exp, stim_nr)
filter_length = st.filter_length

spikes = st.allspikes()
bgsteps = st.bgsteps

# stimulus = asc.rolling_window(bgsteps, filter_length,
#                               preserve_dim=True)
stimulus = packdims(bgsteps, filter_length)
spikes = packdims(spikes, filter_length)


cca = CCA(n_components=n_components,
          scale=True, max_iter=500, tol=1e-06, copy=True)

cca.fit(spikes, stimulus)

x, y = cca.transform(spikes, stimulus)
# x, y = x, y


#%%
import matplotlib.pyplot as plt

cells = cca.x_weights_.T
cells = cells.reshape((n_components, st.nclusters, filter_length))

motionfilt_x = cca.y_weights_[:filter_length].T
motionfilt_y = cca.y_weights_[filter_length:].T
#%%
nsubplots = plf.numsubplots(n_components)
height_ratios = nsubplots[0] * [1, 4]
fig, axes = plt.subplots(nrows=nsubplots[0]*2, ncols=nsubplots[1],
                         gridspec_kw={'height_ratios':height_ratios},
                         figsize=(9,5))

for row, ax_row in enumerate(axes):
    for col, ax in enumerate(ax_row):
        mode_i = (row // (axes.shape[0] // 2)*axes.shape[1] + col)
        if not row % 2:
            ax.plot(motionfilt_x[mode_i, :])
            ax.plot(motionfilt_y[mode_i, :])
            ax.set_xlabel('Time [bins]')
            ax.set_ylabel('Motion magnitude')
            ax.set_ylim(cca.y_weights_.min(), cca.y_weights_.max())
        else:
            im = ax.imshow(cells[mode_i, :], cmap='RdBu_r',
                           vmin=asc.absmin(cells), vmax=asc.absmax(cells),
                         interpolation='none')
            ax.set_ylabel('Cells')
        if mode_i == n_components-1:
            plf.colorbar(im)
plt.show()

#%%

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
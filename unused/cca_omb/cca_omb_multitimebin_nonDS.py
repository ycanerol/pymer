"""

"""
import numpy as np
from sklearn.cross_decomposition import CCA

import matplotlib.pyplot as plt

from omb import OMB
import analysis_scripts as asc
import plotfuncs as plf

from model_fitting_tools import packdims, shiftspikes, cart2pol

exp, stim_nr = '20180710_kilosorted', 8
n_components = 4


st = OMB(exp, stim_nr)
filter_length = st.filter_length

spikes = st.allspikes()
bgsteps = st.bgsteps

# Isolate a single cell
ds_cells = [8, 33, 61, 73, 79]
# Nonds cells
selected_cells = list(set(range(st.nclusters)).difference(set(ds_cells)))
if len(selected_cells) == 1:
    singlecell = True
else:
    singlecell = False
spikes = spikes[selected_cells, :]
st.nclusters = len(selected_cells)
if n_components > st.nclusters:
    n_components = st.nclusters

for shift in [0]:
    print(shift)

    # spikes = shiftspikes(st.allspikes(), shift)

    stimulus = packdims(st.bgsteps, filter_length)
    spikes = packdims(spikes, filter_length)


    cca = CCA(n_components=n_components,
              scale=True, max_iter=500, tol=1e-06, copy=True)

    cca.fit(spikes, stimulus)

    x, y = cca.transform(spikes, stimulus)
    # x, y = x, y


    cells = cca.x_weights_.T
    cells = cells.reshape((n_components, st.nclusters, filter_length))

    motionfilt_x = cca.y_weights_[:filter_length].T
    motionfilt_y = cca.y_weights_[filter_length:].T

    motionfilt_r, motionfilt_theta = cart2pol(motionfilt_x, motionfilt_y)
#%%
    nsubplots = plf.numsubplots(n_components)
    if singlecell:
        nsubplots = (1, 1)
    height_list = [1, 1, 1, 3] # ratios of the plots in each component
    nsubrows = len(height_list)
    height_ratios = nsubplots[0] * height_list
    fig, axes = plt.subplots(nrows=nsubplots[0]*nsubrows, ncols=nsubplots[1],
                             gridspec_kw={'height_ratios':height_ratios},
                             figsize=(9,10))
    if singlecell:
        axes = axes[:, None]
    for row, ax_row in enumerate(axes):
        for col, ax in enumerate(ax_row):
            mode_i = int( row / nsubrows ) * nsubplots[1]  + col
            if singlecell: mode_i -= 1
            if row % nsubrows == 0:
                ax.plot(motionfilt_x[mode_i, :])
                ax.plot(motionfilt_y[mode_i, :])
                if col==0: ax.set_ylabel('Motion')
                ax.set_ylim(cca.y_weights_.min(), cca.y_weights_.max())
                ax.set_title(f'Component {mode_i}', fontweight='bold')
                ax.xaxis.set_ticklabels([])
            elif row % nsubrows == 1:
                ax.plot(motionfilt_r[mode_i, :], color='k')
                if col==0: ax.set_ylabel('Magnitude')
                ax.set_ylim(motionfilt_r.min(), motionfilt_r.max())
                ax.xaxis.set_ticklabels([])
            elif row % nsubrows == 2:
                ax.plot(motionfilt_theta[mode_i, :], color='r')
                ax.yaxis.set_ticks([-np.pi, 0, np.pi])
                ax.yaxis.set_ticklabels(['-π', 0, 'π'])
                ax.xaxis.set_ticklabels([])
            elif row % nsubrows == nsubrows-1:
                im = ax.imshow(cells[mode_i, :], cmap='RdBu_r',
                               vmin=asc.absmin(cells), vmax=asc.absmax(cells),
                               aspect='auto',
                             interpolation='nearest')
                ax.set_xticks(np.array([0, .25, .5, .75, 1]) * cells.shape[-1])
                ax.xaxis.set_ticklabels(np.round((ax.get_xticks()*st.frame_duration), 1))
                ax.set_xlabel('Time [s]')
                if col==0: ax.set_ylabel('Cells')
            if mode_i == n_components-1:
                plf.colorbar(im)
    # fig.tight_layout()
    fig.subplots_adjust(wspace=0.3)
    fig.savefig(f'/home/ycan/Downloads/2020-05-25_labmeeting/cca_nonDScells.pdf')
    plt.show()

#%%

    # im = plt.imshow(cells, cmap='RdBu_r', vmin=asc.absmin(cells), vmax=asc.absmax(cells))
    # plt.ylabel('Components')
    # plt.xlabel('Cells')
    # plf.colorbar(im)
    # plt.show()



    plt.plot(cca.coef_[:, 0], cca.coef_[:, 1], 'ko')
    plt.show()

    offsets = np.arange(n_components)*.6
    plt.plot(cca.y_weights_ + np.tile(offsets, (stimulus.shape[1], 1)), lw=0.5)
    plt.hlines(offsets, 0, stimulus.shape[1], lw=.2)

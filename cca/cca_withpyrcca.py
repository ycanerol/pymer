
import numpy as np
import matplotlib.pyplot as plt
import rcca

import analysis_scripts as asc
import model_fitting_tools as mft
import spikeshuffler
import plotfuncs as plf
from omb import OMB


exp, stim_nr = '20180710_kilosorted', 8
n_components = 20
shufflespikes = False

savedir = '/Users/ycan/Downloads/2020-06-10_meeting/postmeeting/'
savefig = True

cca = rcca.CCA(kernelcca=False, reg=0.01, numCC=n_components)

st = OMB(exp, stim_nr)
filter_length = st.filter_length

spikes = st.allspikes()
bgsteps = st.bgsteps

nspikes_percell = spikes.sum(axis=1)

if shufflespikes:
    spikes = spikeshuffler.shufflebyrow(spikes)

#sp_train, sp_test, stim_train, stim_test = train_test_split(spikes, bgsteps)

stimulus = mft.packdims(st.bgsteps, filter_length)
spikes = mft.packdims(spikes, filter_length)

cca.train([spikes, stimulus])

cells = np.swapaxes(cca.ws[0], 1, 0)
cells = cells.reshape((n_components, st.nclusters, filter_length))

cells_sorted_nsp = cells[:, np.argsort(nspikes_percell), :]

motionfilt_x = cca.ws[1][:filter_length].T
motionfilt_y = cca.ws[1][filter_length:].T

motionfilt_r, motionfilt_theta = mft.cart2pol(motionfilt_x, motionfilt_y)
#%%
nrows, ncols = plf.numsubplots(n_components)
fig_cells, axes_cells = plt.subplots(nrows, ncols, figsize=(10, 10))

for i in range(n_components):
    ax = axes_cells.flat[i]
    im = ax.imshow(cells[i, :], cmap='RdBu_r',
                    vmin=asc.absmin(cells),
                    vmax=asc.absmax(cells),
                    aspect='auto', interpolation='nearest')
    ax.set_title(f'{i}')
fig_cells.suptitle(f'Cells default order {shufflespikes=}')
if savefig:
    fig_cells.savefig(savedir + f'{n_components=}_{shufflespikes=}_cells_default_order.pdf')
# plt.show()
plt.close(fig_cells)
#%%
# fig_cells_sorted, axes_cells_sorted = plt.subplots(nrows, ncols, figsize=(10, 10))
# for i in range(n_components):
#     ax = axes_cells_sorted.flat[i]
#     im = ax.imshow(cells_sorted_nsp[i, :], cmap='RdBu_r',
#                     vmin=asc.absmin(cells_sorted_nsp),
#                     vmax=asc.absmax(cells_sorted_nsp),
#                     aspect='auto', interpolation='nearest')
#     ax.set_title(f'{i}')
# fig_cells_sorted.suptitle(f'Cells sorted by number of sp {shufflespikes=}')
# plt.show()
# plt.close(fig_cells_sorted)
#%%

nsubplots = plf.numsubplots(n_components)
# nsubplots = plf.numsubplots(3)
height_list = [1, 1, 1, 3] # ratios of the plots in each component
nsubrows = len(height_list)
height_ratios = nsubplots[0] * height_list
fig, axes = plt.subplots(nrows=nsubplots[0]*nsubrows, ncols=nsubplots[1],
                         gridspec_kw={'height_ratios':height_ratios},
                         # sharey=True,
                         figsize=(9,10))

for row, ax_row in enumerate(axes):
    for col, ax in enumerate(ax_row):
        mode_i = int( row / nsubrows ) * nsubplots[1]  + col
        # ax.text(0.5, 0.5, f'{mode_i}')
        ax.set_yticks([])
        if row % nsubrows == 0:
            ax.plot(motionfilt_x[mode_i, :])
            ax.plot(motionfilt_y[mode_i, :])
            if col==0: ax.set_ylabel('Motion', rotation=0, ha='right', va='center')
            ax.set_ylim(cca.ws[1].min(), cca.ws[1].max())
            # ax.set_title(f'Component {mode_i}', fontweight='bold')
            ax.xaxis.set_ticklabels([])
        elif row % nsubrows == 1:
            ax.plot(motionfilt_r[mode_i, :], color='k')
            if col==0: ax.set_ylabel('Magnitude', rotation=0, ha='right', va='center')
            ax.set_ylim(motionfilt_r.min(), motionfilt_r.max())
            ax.xaxis.set_ticklabels([])
        elif row % nsubrows == 2:
            ax.plot(motionfilt_theta[mode_i, :], color='r')
            if mode_i == 0:
                ax.yaxis.set_ticks([-np.pi, 0, np.pi])
                ax.yaxis.set_ticklabels(['-π', 0, 'π'])
            ax.xaxis.set_ticklabels([])
        elif row % nsubrows == nsubrows-1:
            # im = ax.imshow(cells[mode_i, :], cmap='RdBu_r',
            im = ax.imshow(cells_sorted_nsp[mode_i, :], cmap='RdBu_r',
                           vmin=asc.absmin(cells), vmax=asc.absmax(cells),
                           aspect='auto',
                         interpolation='nearest')
            ax.set_xticks([])
            if row == axes.shape[0]-1:
                ax.set_xlabel('Time [s]')
                ax.set_xticks(np.array([0, .25, .5, .75, 1]) * cells.shape[-1])
                ax.xaxis.set_ticklabels(np.round((ax.get_xticks()*st.frame_duration), 1))
            if col==0: ax.set_ylabel('Cells\n(sorted nsp)', rotation=0, ha='right', va='center')

            if mode_i == n_components-1:
                plf.colorbar(im)
# fig.tight_layout()
fig.suptitle(f'CCA components of {st.exp_foldername}\n{shufflespikes=} {n_components=}\n{}')
fig.subplots_adjust(wspace=0.1)
if savefig:
    fig.savefig(savedir + f'{n_components=}_{shufflespikes=}_cellsandcomponents.pdf')
plt.show()
# plt.close(fig)

#%%
fig_corrs = plt.figure()
plt.plot(cca.cancorrs, 'ko')
plt.ylim([0.17, 0.24])
plt.xlabel('Component index')
plt.ylabel('Correlation')
plt.title(f'Cannonical correlations {shufflespikes=}')
if savefig:
    fig_corrs.savefig(savedir + f'{n_components=}_{shufflespikes=}_correlation_coeffs.pdf')
# plt.show()
plt.close(fig_corrs)


import numpy as np
import matplotlib.pyplot as plt
import rcca

import analysis_scripts as asc
import model_fitting_tools as mft
import spikeshuffler
import plotfuncs as plf
from omb import OMB


exp, stim_nr = '20180710_kilosorted', 8
n_components = 6
shufflespikes = True

savedir = '/Users/ycan/Downloads/2020-06-10_meeting/'
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

motionfilt_x = cca.ws[1][:filter_length]
motionfilt_y = cca.ws[1][filter_length:]

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
    fig_cells.savefig(savedir + f'cells_{n_components=}_{shufflespikes=}_default_order.pdf')
plt.show()
#%%
fig_cells_sorted, axes_cells_sorted = plt.subplots(nrows, ncols, figsize=(10, 10))
for i in range(n_components):
    ax = axes_cells_sorted.flat[i]
    im = ax.imshow(cells_sorted_nsp[i, :], cmap='RdBu_r',
                    vmin=asc.absmin(cells_sorted_nsp),
                    vmax=asc.absmax(cells_sorted_nsp),
                    aspect='auto', interpolation='nearest')
    ax.set_title(f'{i}')
fig_cells_sorted.suptitle(f'Cells sorted by number of sp {shufflespikes=}')
plt.show()


#%%
fig_stimfilters, axes_stimfilters = plt.subplots(2, 2, figsize=(9, 7))
(ax_x, ax_y, ax_r, ax_th) = axes_stimfilters.flat
ax_x.plot(motionfilt_x)
ax_y.plot(motionfilt_y)
ax_r.plot(motionfilt_r)
ax_th.plot(motionfilt_theta)
if savefig:
    fig_stimfilters.savefig(savedir + f'stimcomponents_{n_components=}_{shufflespikes=}.pdf')
plt.show()
#%%
fig_corrs = plt.figure()
plt.plot(cca.cancorrs, 'ko')
plt.ylim([0.17, 0.24])
plt.xlabel('Component index')
plt.ylabel('Correlation')
plt.title(f'Cannonical correlations {shufflespikes=}')
if savefig:
    fig_corrs.savefig(savedir + f'Correlations_{n_components=}_{shufflespikes=}.pdf')
plt.show()

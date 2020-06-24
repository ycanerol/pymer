
import numpy as np
import matplotlib.pyplot as plt
import rcca

import analysis_scripts as asc
import model_fitting_tools as mft
import spikeshuffler
import plotfuncs as plf
from omb import OMB


def cca_omb_components(exp: str, stim_nr: int,
                       n_components: int = 10,
                       shufflespikes: bool = False, savedir: str = None,
                       savefig: bool = True, sort_by_nspikes: bool = True,
                       select_cells: list = None,
                       plot_first_ncells: int = None):

    """
    Analyze OMB responses using cannonical correlation analysis and plot the results.

    Parameters
    ---
    n_components:
        Number of components that will be requested from the CCA anaylsis. More numbers mean
        the algortihm will stop at a later point. That means components of analyses with fewer
        n_components are going to be identical to the first n components of the higher-number
        component analyses.
    shufflespikes: bool
        Whether to randomize the spikes, to validate the results
    savedir: str
        Custom directory to save the figures and data files. If None, will be saved in the experiment
        directory under appropritate path.
    savefig: bool
        Whether to save the figures.
    sort_by_nspikes: bool
        Wheter to sort the cell weights array by the number of spikes during the stimulus.
    select_cells: list
       A list of indexes for the subset of cells to perform the analysis for.
    plot_first_ncells: int
        Number of cells to plot in the cell plots.
    """


    cca = rcca.CCA(kernelcca=False, reg=0.01, numCC=n_components)

    st = OMB(exp, stim_nr)
    filter_length = st.filter_length

    if savedir is None:
        savedir = st.stim_dir / 'CCA'
        savedir.mkdir(exist_ok=True)

    spikes = st.allspikes()
    bgsteps = st.bgsteps

    if select_cells is not None:
        spikes = spikes[select_cells]
        st.nclusters = len(select_cells)

    nspikes_percell = spikes.sum(axis=1)

    if shufflespikes:
        spikes = spikeshuffler.shufflebyrow(spikes)

    #sp_train, sp_test, stim_train, stim_test = train_test_split(spikes, bgsteps)

    stimulus = mft.packdims(st.bgsteps, filter_length)
    spikes = mft.packdims(spikes, filter_length)

    cca.train([spikes, stimulus])

    cells = np.swapaxes(cca.ws[0], 1, 0)
    cells = cells.reshape((n_components, st.nclusters, filter_length))

    nsp_argsorted = np.argsort(nspikes_percell)
    cells_sorted_nsp = cells[:, nsp_argsorted, :]

    if sort_by_nspikes:
        cells_toplot = cells_sorted_nsp
    else:
        cells_toplot = cells

    if plot_first_ncells is not None:
        cells_toplot = cells_toplot[:, :plot_first_ncells, ...]

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
        fig_cells.savefig(savedir / f'{n_components=}_{shufflespikes=}_cells_default_order.pdf')
    plt.close(fig_cells)

    nsubplots = plf.numsubplots(n_components)
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
                im = ax.imshow(cells_toplot[mode_i, :], cmap='RdBu_r',
                               vmin=asc.absmin(cells), vmax=asc.absmax(cells),
                               aspect='auto',
                             interpolation='nearest')
                ax.set_xticks([])
                if row == axes.shape[0]-1:
                    ax.set_xlabel('Time [s]')
                    ax.set_xticks(np.array([0, .25, .5, .75, 1]) * cells_toplot.shape[-1])
                    ax.xaxis.set_ticklabels(np.round((ax.get_xticks()*st.frame_duration), 1))
                if col==0: ax.set_ylabel(f'Cells\n{"(sorted nsp)"*sort_by_nspikes}', rotation=0, ha='right', va='center')

                if mode_i == n_components-1:
                    plf.colorbar(im)
    # fig.tight_layout()
    fig.suptitle(f'CCA components of {st.exp_foldername}\n{shufflespikes=} {n_components=}\n{sort_by_nspikes=}')
    fig.subplots_adjust(wspace=0.1)
    if savefig:
        fig.savefig(savedir / f'{n_components=}_{shufflespikes=}_cellsandcomponents.pdf')
    # plt.show()
    plt.close(fig)

    #%%
    fig_corrs = plt.figure()
    plt.plot(cca.cancorrs, 'ko')
    plt.ylim([0.17, 0.24])
    plt.xlabel('Component index')
    plt.ylabel('Correlation')
    plt.title(f'Cannonical correlations {shufflespikes=}')
    if savefig:
        fig_corrs.savefig(savedir / f'{n_components=}_{shufflespikes=}_correlation_coeffs.pdf')
    # plt.show()
    plt.close(fig_corrs)


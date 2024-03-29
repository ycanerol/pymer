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

from cca_macke_function import cca_macke
from rcca_omb import cca_rcca


def cca_omb_components(exp: str, stim_nr: int,
                       n_components: int = 6,
                       regularization=None,
                       filter_length=None,
                       cca_solver: str = 'macke',
                       maxframes=None,
                       shufflespikes: bool = False,
                       exclude_allzero_spike_rows: bool = True,
                       savedir: str = None,
                       savefig: bool = True, sort_by_nspikes: bool = True,
                       select_cells: list = None,
                       plot_first_ncells: int = None,
                       whiten:bool = False):

    """
    Analyze OMB responses using cannonical correlation analysis and plot the results.

    Parameters
    ---
    n_components:
        Number of components that will be requested from the CCA anaylsis. More numbers mean
        the algortihm will stop at a later point. That means components of analyses with fewer
        n_components are going to be identical to the first n components of the higher-number
        component analyses.
    regularization:
        The regularization parameter to be passed onto rcca.CCA. Not relevant for macke
    filter_length:
        The length of the time window to be considered in the past for the stimulus and the responses.
        Can be different for stimulus and response, if a tuple is given.
    cca_solver:
        Which CCA solver to use. Options are `rcca` and `macke`(default)
    maxframes: int
        Number of frames to load in the the experiment object. Used to avoid memory and performance
        issues.
    shufflespikes: bool
        Whether to randomize the spikes, to validate the results
    exclude_allzero_spike_rows:
        Exclude all cells which have zero spikes for the duration of the stimulus.
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
    if regularization is None:
        regularization = 0

    st = OMB(exp, stim_nr, maxframes=maxframes)

    if filter_length is None:
        filter_length = st.filter_length

    if type(filter_length) is int:
        filter_length = (filter_length, filter_length)

    if type(savedir) is str:
        savedir = Path(savedir)

    if savedir is None:
        savedir = st.stim_dir / 'CCA'
        savedir.mkdir(exist_ok=True, parents=True)

    spikes = st.allspikes()
    nonzerospikerows = ~np.isclose(spikes.sum(axis=1), 0)
    # Set the mean to zero for spikes
    spikes -= spikes.mean(axis=1)[:, None]

    bgsteps = st.bgsteps

    if select_cells is not None:
        if type(select_cells) is not np.array:
            select_cells = np.array(select_cells)
        spikes = spikes[select_cells]
        st.nclusters = len(select_cells)
        # Convert to list for better string representation
        # np.array is printed as "array([....])"
        # with newline characters which is problematic in filenames
        select_cells = list(select_cells)

    # Exclude rows that have no spikes throughout
    if exclude_allzero_spike_rows:
        spikes = spikes[nonzerospikerows, :]

    nspikes_percell = spikes.sum(axis=1)

    if shufflespikes:
        spikes = spikeshuffler.shufflebyrow(spikes)

    figsavename = f'{n_components=}_{shufflespikes=}_{select_cells=}_{regularization=}_{filter_length=}_{whiten=}'
    # If the file length gets too long due to the list of selected cells, summarize it.
    if len(figsavename) > 200:
        figsavename = f'{n_components=}_{shufflespikes=}_select_cells={len(select_cells)}cells-index{select_cells[0]}to{select_cells[-1]}_{regularization=}_{filter_length=}_{whiten=}'

    #sp_train, sp_test, stim_train, stim_test = train_test_split(spikes, bgsteps)

    stimulus = mft.packdims(st.bgsteps, filter_length[0])
    spikes = mft.packdims(spikes, filter_length[1])


    if cca_solver == 'rcca':
        resp_comps, stim_comps, cancorrs = cca_rcca(spikes, stimulus, filter_length,
                                        n_components, regularization, whiten)

        # cells = np.swapaxes(spikes_res, 1, 0)
        # cells = cells.reshape((n_components, st.nclusters, filter_length[1]))
        # motionfilt_x = cca.ws[1][:filter_length[0]].T
        # motionfilt_y = cca.ws[1][filter_length[0]:].T
    elif cca_solver == 'macke':
        resp_comps, stim_comps, cancorrs = cca_macke(spikes, stimulus,
                                                    filter_length, n_components)

    nsp_argsorted = np.argsort(nspikes_percell)
    resp_comps_sorted_nsp = resp_comps[:, nsp_argsorted, :]

    if sort_by_nspikes:
        resp_comps_toplot = resp_comps_sorted_nsp
    else:
        resp_comps_toplot = resp_comps

    if plot_first_ncells is not None:
        resp_comps_toplot = resp_comps_toplot[:, :plot_first_ncells, ...]


    motionfilt_r, motionfilt_theta = mft.cart2pol(stim_comps[:, 0, :],
                                                  stim_comps[:, 1, :])
    #%%
    nrows, ncols = plf.numsubplots(n_components)
    fig_cells, axes_cells = plt.subplots(nrows, ncols, figsize=(10, 10))

    for i in range(n_components):
        ax = axes_cells.flat[i]
        im = ax.imshow(resp_comps[i, :], cmap='RdBu_r',
                        vmin=asc.absmin(resp_comps),
                        vmax=asc.absmax(resp_comps),
                        aspect='auto', interpolation='nearest')
        ax.set_title(f'{i}')
    fig_cells.suptitle(f'Cells default order {shufflespikes=}')
    if savefig:
        fig_cells.savefig(savedir / f'{figsavename}_cells_default_order.pdf')
    plt.close(fig_cells)

    nsubplots = plf.numsubplots(n_components)
    height_list = [1, 1, 1, 3] # ratios of the plots in each component

    # Create a time vector for the stimulus plots
    t_stim = -np.arange(0, filter_length[0]*st.frame_duration, st.frame_duration)[::-1] * 1000
    t_response = -np.arange(0, filter_length[1]*st.frame_duration, st.frame_duration)[::-1] * 1000
    xtick_loc_params = dict(nbins=4, steps=[2, 5, 10], integer=True)

    nsubrows = len(height_list)
    height_ratios = nsubplots[0] * height_list
    fig, axes = plt.subplots(nrows=nsubplots[0]*nsubrows, ncols=nsubplots[1],
                             gridspec_kw={'height_ratios':height_ratios},
                             figsize=(11,10))

    for row, ax_row in enumerate(axes):
        for col, ax in enumerate(ax_row):
            mode_i = int( row / nsubrows ) * nsubplots[1]  + col
            # ax.text(0.5, 0.5, f'{mode_i}')
            ax.set_yticks([])
            # Plot motion filters
            if row % nsubrows == 0:

                ax.plot(t_stim, stim_comps[mode_i, 0, :], marker='o', markersize=1)
                ax.plot(t_stim, stim_comps[mode_i, 1, :], marker='o', markersize=1)
                if col==0: ax.set_ylabel('Motion', rotation=0, ha='right', va='center')
                ax.set_ylim(stim_comps.min(), stim_comps.max())

                # Draw a horizontal line for zero and prevent rescaling of x-axis
                xlims = ax.get_xlim()
                ax.hlines(0, *ax.get_xlim(), colors='k', linestyles='dashed', alpha=0.3)
                ax.set_xlim(*xlims)

                # ax.set_title(f'Component {mode_i}', fontweight='bold')

                ax.xaxis.set_major_locator(MaxNLocator(**xtick_loc_params))

                if not mode_i == 0 or filter_length[0] == filter_length[1]:
                    ax.xaxis.set_ticklabels([])
                else:
                    ax.tick_params(axis='x', labelsize=8)

            # Plot magnitude of motion
            elif row % nsubrows == 1:
                ax.plot(t_stim, motionfilt_r[mode_i, :], color='k', marker='o', markersize=1)
                if col==0: ax.set_ylabel('Magnitude', rotation=0, ha='right', va='center')
                ax.set_ylim(motionfilt_r.min(), motionfilt_r.max())
                ax.xaxis.set_ticklabels([])
                ax.xaxis.set_major_locator(MaxNLocator(**xtick_loc_params))
            # Plot direction of motion
            elif row % nsubrows == 2:
                ax.plot(t_stim, motionfilt_theta[mode_i, :], color='r', marker='o', markersize=1)
                if mode_i == 0:
                    ax.yaxis.set_ticks([-np.pi, 0, np.pi])
                    ax.yaxis.set_ticklabels(['-π', 0, 'π'])
                ax.xaxis.set_ticklabels([])
                ax.xaxis.set_major_locator(MaxNLocator(**xtick_loc_params))
            # Plot cell weights
            elif row % nsubrows == nsubrows-1:
                im = ax.imshow(resp_comps_toplot[mode_i, :], cmap='RdBu_r',
                               vmin=asc.absmin(resp_comps), vmax=asc.absmax(resp_comps),
                               aspect='auto',
                               interpolation='nearest',
                               extent=[t_response[0], t_response[-1], 0, resp_comps_toplot.shape[1]])
                ax.xaxis.set_major_locator(MaxNLocator(**xtick_loc_params))
                if row == axes.shape[0]-1:
                    ax.set_xlabel('Time before spike [ms]')
                    # ax.set_xticks(np.array([0, .25, .5, .75, 1]) * cells_toplot.shape[-1])
                    # ax.xaxis.set_ticklabels(-np.round((ax.get_xticks()*st.frame_duration), 2)[::-1])
                else:
                    ax.xaxis.set_ticklabels([])

                plf.integerticks(ax, 5, which='y')
                if col==0:
                    ax.set_ylabel(f'Cells\n{"(sorted nsp)"*sort_by_nspikes}\n{("(first " + str(plot_first_ncells)+ " cells)")*(type(plot_first_ncells) is int) }',
                                  rotation=0, ha='right', va='center')
                else:
                    ax.yaxis.set_ticklabels([])
                if mode_i == n_components-1:
                    plf.colorbar(im)
            # Add ticks on the right side of the plots
            if col == nsubplots[1]-1 and row % nsubrows != nsubrows - 1:
                plf.integerticks(ax, 3, which='y')
                ax.yaxis.tick_right()

    fig.suptitle(f'CCA components of {st.exp_foldername}\n{shufflespikes=} {n_components=}\n{sort_by_nspikes=}\n'
            + f'{select_cells=} {regularization=} {filter_length=}')
    fig.subplots_adjust(wspace=0.1, hspace=0.3)
    if savefig:
        fig.savefig(savedir / f'{figsavename}_cellsandcomponents.pdf')
    # plt.show()
    plt.close(fig)

    #%%
    fig_corrs = plt.figure()
    plt.plot(cancorrs, 'ko')
    # plt.ylim([0.17, 0.24])
    plt.xlabel('Component index')
    plt.ylabel('Correlation')
    plt.title(f'Cannonical correlations {shufflespikes=}')
    if savefig:
        fig_corrs.savefig(savedir / f'{figsavename}_correlation_coeffs.pdf')
    # plt.show()
    plt.close(fig_corrs)

    fig_nlt, axes_nlt = plt.subplots(nrows, ncols, figsize=(10, 10))

    stim_comps_flatter = stim_comps[:n_components].reshape((n_components, 2*filter_length[0])).T
    resp_comps_flatter = resp_comps[:n_components].reshape((n_components, resp_comps.shape[1]*filter_length[1])).T

    # from IPython.core.debugger import Pdb; ipdb=Pdb(); ipdb.set_trace()
    # Reshape to perform the convolution as a matrix multiplication
    generator_stim = stimulus @ stim_comps_flatter
    generator_resp = spikes @ resp_comps_flatter


    for i, ax in enumerate(axes_nlt.flatten()):


        nonlinearity, bins = nlt.calc_nonlin(generator_resp[:, i], generator_stim[:, i])
        # ax.scatter(generator_stim, generator_resp, s=1, alpha=0.5, facecolor='grey')
        ax.plot(bins, nonlinearity, 'k')
        if i == 0:
            all_nonlinearities = np.empty((n_components, *nonlinearity.shape))
            all_bins = np.empty((n_components, *bins.shape))
        all_nonlinearities[i, ...] = nonlinearity
        all_bins[i, ...] = bins

    nlt_xlims = []
    nlt_ylims = []
    for i, ax in enumerate(axes_nlt.flatten()):
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        nlt_xlims.extend(xlim)
        nlt_ylims.extend(ylim)
    nlt_maxx, nlt_minx = max(nlt_xlims), min(nlt_xlims)
    nlt_maxy, nlt_miny = max(nlt_ylims), min(nlt_ylims)

    for i, ax in enumerate(axes_nlt.flatten()):
        ax.set_xlim([nlt_minx, nlt_maxx])
        ax.set_ylim([nlt_miny, nlt_maxy])

    for i, axes_row in enumerate(axes_nlt):
        for j, ax in enumerate(axes_row):
            if i == nrows-1:
                ax.set_xlabel('Generator (motion)')
            if j == 0:
                ax.set_ylabel('Generator (cells)')
            else:
                ax.yaxis.set_ticklabels([])
            ax.set_xlim([nlt_minx, nlt_maxx])
            ax.set_ylim([nlt_miny, nlt_maxy])

    fig_nlt.suptitle(f'Nonlinearities\n{figsavename}')
    if savefig:
        fig_nlt.savefig(savedir / f'{figsavename}_nonlinearity.png')
    plt.close(fig_nlt)
    keystosave = [ 'n_components', 'resp_comps', 'stim_comps',
                    'motionfilt_r', 'motionfilt_theta', 'resp_comps_sorted_nsp',
                    'select_cells', 'regularization', 'filter_length',
                    'all_nonlinearities', 'all_bins', 'cca_solver']
    datadict = dict()
    for key in keystosave:
        datadict[key] = locals()[key]
    np.savez(savedir / figsavename, **datadict)


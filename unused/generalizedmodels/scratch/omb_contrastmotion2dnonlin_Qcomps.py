#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Calculate 2D nonlinearity between motion and contrast
to see whether only contrast is triggering the spikes
or if it's a combination of motion and contrast

"""

import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gsp

import plotfuncs as plf
import nonlinearity as nlt
from omb import OMB

import gen_quad_model_multidimensional as gqm


def omb_contrastmotion2dnonlin_Qcomps(exp, stim, nbins_nlt=9, cmap='Greys'):
    """
    Calculate and plot the 2D nonlinearities for the OMB stimulus. Multiple
    components of the matrix Q for the motion.

    Parameters:
    ------
        nbins_nlt:
            Number of bins to be used for dividing the generator signals
            into ranges with equal number of samples.
    """

    st = OMB(exp, stim)

    # Motion and contrast
    data_cm = np.load(os.path.join(st.exp_dir, 'data_analysis',
                                   st.stimname, 'GQM_motioncontrast_val',
                                   f'{stim}_GQM_motioncontrast_val.npz'))

    qall = data_cm['Qall']
    kall = data_cm['kall']
    muall = data_cm['muall']

    eigvecs = data_cm['eigvecs']
    eigvals = data_cm['eigvals']

    eiginds = [0, 1, st.filter_length-2, st.filter_length-1]
    # HINT
    eiginds = [eiginds[-1]]
    eiginds = [0]

    cross_corrs = data_cm['cross_corrs']

    allspikes = st.allspikes()

    stim_mot = st.bgsteps.copy()

    # Bin dimension should be one greater than nonlinearity for pcolormesh
    # compatibility. Otherwise the last row and column of nonlinearity is not
    # plotted.
    all_bins_c = np.zeros((st.nclusters, nbins_nlt+1))
    all_bins_r = np.zeros((st.nclusters, nbins_nlt+1))
    nonlinearities = np.zeros((st.nclusters, nbins_nlt, nbins_nlt))

    label = '2D-nonlin_compQ_motion_kcontrast'

    savedir = os.path.join(st.stim_dir, label)
    os.makedirs(savedir, exist_ok=True)

    for i in range(st.nclusters):
        stim_con = st.contrast_signal_cell(i).squeeze()

        for eigind in eiginds:
            generator_x = np.convolve(eigvecs[i, 0, :, eigind],
                                      stim_mot[0, :], 'full')[:-st.filter_length+1]
            generator_y = np.convolve(eigvecs[i, 1, :, eigind],
                                      stim_mot[1, :], 'full')[:-st.filter_length+1]

        # Calculate the magnitude of the vector formed by motion generators
        generators = np.vstack([generator_x, generator_y])
        r = np.sqrt(np.sum(generators**2, axis=0))

        # Project the contrast stimulus onto the linear filter
        generator_c = np.convolve(stim_con,
                                  kall[i, 2, :],
                                  'full')[:-st.filter_length+1]
        spikes = allspikes[i, :]

        nonlinearity, bins_c, bins_r = nlt.calc_nonlin_2d(spikes,
                                                          generator_c,
                                                          r, nr_bins=nbins_nlt)
        nonlinearity /= st.frame_duration

        all_bins_c[i, :] = bins_c
        all_bins_r[i, :] = bins_r
        nonlinearities[i, ...] = nonlinearity

        X, Y = np.meshgrid(bins_c, bins_r, indexing='ij')

        fig = plt.figure()

        gs = gsp.GridSpec(5, 5)
        axmain = plt.subplot(gs[1:, :-1])
        axx = plt.subplot(gs[0, :-1], sharex=axmain)
        axy = plt.subplot(gs[1:, -1], sharey=axmain)

        # Normally subplots turns off shared axis tick labels but
        # Gridspec does not do this
        plt.setp(axx.get_xticklabels(), visible=False)
        plt.setp(axy.get_yticklabels(), visible=False)

        im = axmain.pcolormesh(X, Y, nonlinearity, cmap=cmap)
        plf.integerticks(axmain)

        cb = plt.colorbar(im)
        cb.outline.set_linewidth(0)
        cb.ax.set_xlabel('spikes/s')
        cb.ax.xaxis.set_label_position('top')

        plf.integerticks(cb.ax, 4, which='y')
        plf.integerticks(axx, 1, which='y')
        plf.integerticks(axy, 1, which='x')

        barkwargs = dict(alpha=.3, facecolor='k',
                         linewidth=.5, edgecolor='w')

        axx.bar(nlt.bin_midpoints(bins_c), nonlinearity.mean(axis=1),
                width=np.ediff1d(bins_c), **barkwargs)
        axy.barh(nlt.bin_midpoints(bins_r), nonlinearity.mean(axis=0),
                 height=np.ediff1d(bins_r), **barkwargs)
        plf.spineless(axx, 'b')
        plf.spineless(axy, 'l')

        axmain.set_xlabel('Projection onto linear contrast filter')
        axmain.set_ylabel(f'Magnitude of projection onto Q components {eigind}')
        fig.suptitle(f'{st.exp_foldername}\n{st.stimname}\n{st.clids[i]} '
                     f'2D nonlinearity nsp: {st.allspikes()[i, :].sum():<5.0f}')

        plt.subplots_adjust(top=.85)
        fig.savefig(os.path.join(savedir, st.clids[i]), bbox_inches='tight')
        plt.show()

    keystosave = ['nonlinearities', 'all_bins_c', 'all_bins_r', 'nbins_nlt']
    datadict = {}

    for key in keystosave:
        datadict.update({key: locals()[key]})
    npzfpath = os.path.join(savedir, f'{st.stimnr}_{label}.npz')
    np.savez(npzfpath, **datadict)

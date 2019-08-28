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

import plotfuncs as plf
import nonlinearity as nlt
from omb import OMB

import gen_quad_model_multidimensional as gqm

from scipy import stats

exp, stim = '20180710', 8
#exp, stim = 'Kuehn', 13

st = OMB(exp, stim)

# Motion and contrast
data_cm = np.load(os.path.join(st.exp_dir, 'data_analysis',
                               st.stimname, 'GQM_motioncontrast',
                               f'{stim}_GQM_motioncontrast_data.npz'))

qall = data_cm['Qall']
kall = data_cm['kall']
muall = data_cm['muall']
cross_corrs = data_cm['cross_corrs']

allspikes = st.allspikes()

stim_mot = st.bgsteps.copy()

nbins_nlt = 9
cmap = 'Greys'
plot3d = False

all_bins_c = np.zeros((st.nclusters, nbins_nlt))
all_bins_r = np.zeros((st.nclusters, nbins_nlt))
nonlinearities = np.zeros((st.nclusters, nbins_nlt, nbins_nlt))

savedir = os.path.join(st.stim_dir, '2D-nonlin_magQ_motion_kcontrast')
os.makedirs(savedir, exist_ok=True)

for i in range(st.nclusters):
    stim_con = st.contrast_signal_cell(i).squeeze()

    # Project the motion stimulus onto the quadratic filter
    generator_x = gqm.conv2d(qall[i, 0, :], stim_mot[0, :])
    generator_y = gqm.conv2d(qall[i, 1, :], stim_mot[1, :])

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

    X, Y = np.meshgrid(bins_c, bins_r)

    plt.figure()
    im = plt.pcolormesh(X, Y, nonlinearity, cmap=cmap)
    ax = im.axes
    plf.integerticks(ax)
    cb = plf.colorbar(im, title='spikes/s')
    plf.integerticks(cb.ax, which='y')
    ax.set_xlabel('Projection onto linear contrast filter')
    ax.set_ylabel('Magnitude of projection onto quadratic motion filters')
    ax.set_title(f'{st.exp_foldername}\n{st.stimname}\n2D nonlinearity\n'
                 f'{st.clids[i]} nsp: {st.allspikes()[i, :].sum():<5.0f}')
    plt.savefig(os.path.join(savedir, st.clids[i]), bbox_inches='tight')
    plt.show()

    if plot3d:
        if i == 0:
            from mpl_toolkits import mplot3d
        from matplotlib.ticker import MaxNLocator
        #%%
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.plot_surface(X, Y, nonlinearity, cmap='YlGn', edgecolors='k', linewidths=0.2)
        ax.set_xlabel('Projection onto linear contrast filter')
        ax.set_ylabel('Magnitude of projection onto quadratic motion filters')

        ax.set_zlabel(r'Firing rate [sp/s]')
        ax.view_init(elev=30, azim=-135)

        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.zaxis.set_major_locator(MaxNLocator(integer=True))

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Calculate 2D nonlinearity between motion and contrast
to see whether only contrast is triggering the spikes
or if it's a combination of motion and contrast

"""

import sys
import os

import numpy as np
import matplotlib.pyplot as plt

import nonlinearity as nlt
from omb import OMB

import gen_quad_model_multidimensional as gqm

from scipy import stats

exp, stim = '20180710', 8
#exp, stim = 'Kuehn', 13

st = OMB(exp, stim)

# Motion and contrast
data_cm = np.load(os.path.join(st.exp_dir, 'data_analysis',
                               st.stimname, 'GQM_motioncontrast', f'{stim}_GQM_motioncontrast_data.npz'))

qall = data_cm['Qall']
kall = data_cm['kall']
muall = data_cm['muall']
eigvecs = data_cm['eigvecs']
cross_corrs = data_cm['cross_corrs']

allspikes = st.allspikes()

stim_mot = st.bgsteps.copy()
r = np.sqrt(np.sum(stim_mot**2, axis=0))

eigen_index = -1
cmap = 'Greens'

for i in range(st.nclusters):
    stim_con = st.contrast_signal_cell(i).squeeze()

    generator_x = np.convolve(stim_mot[0, :],
                              eigvecs[i, 0, :, eigen_index],
                              'full')[:-st.filter_length+1]
    generator_y = np.convolve(stim_mot[1, :],
                              eigvecs[i, 1, :, eigen_index],
                              'full')[:-st.filter_length+1]

    generators = np.vstack([generator_x, generator_y])
    r = np.sqrt(np.sum(generators**2, axis=0))

    generator_c = np.convolve(stim_con,
                              kall[i, 2, :],
                              'full')[:-st.filter_length+1]

    nonlinearity, bins_c, bins_r = nlt.calc_nonlin_2d(allspikes[i, :], generator_c, r)

#    plt.imshow(nonlinearity, cmap=cmap)
#    plt.show()

    plt.scatter(generator_c, r, s=np.sqrt(allspikes[i, :]), color='k')
    plt.title(i)
    plt.show()

    X, Y = np.meshgrid(bins_c, bins_r)

    plt.figure()
    plt.pcolormesh(X, Y, nonlinearity, cmap=cmap)
    plt.xlabel('Contrast generator strength')
    plt.ylabel('Motion generator strength hypotenuse')
    plt.title(i)
#    plt.axis('equal')
    plt.show()
    break
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  7 14:54:08 2018

@author: ycan
"""
import os
import numpy as np
import matplotlib.pyplot as plt

from randpy import randpy
import analysis_scripts as asc
import plotfuncs as plf
import iofuncs as iof


def calc_covar(stim_small):
    """
    Calculate the covariance matrix for a given 1-D stimulus snippet.
    """
    # To be able to transpose with .T method
    stim = stim_small[np.newaxis, :]
    covar = np.dot(stim.T, stim)
    return covar


def OMBanalyzer(exp_name, stimnr):
    """
    Analyze responses to object moving background stimulus.
    """

    # TODO
    # The stimulus computer sometimes crashes before finishing the stimulus,
    # this case should be handled (e.g. exp_name: '20180712*eye1', stimnr:10)
    # TODO
    # Calculate nonlinearities
    # TODO
    # Add iteration over multiple stimuli

    exp_dir = iof.exp_dir_fixer(exp_name)
    exp_name = os.path.split(exp_dir)[-1]
    stimname = iof.getstimname(exp_dir, stimnr)

    parameters = asc.read_parameters(exp_name, stimnr)
    assert(parameters['stimulus_type']=='objectsmovingbackground')
    stimframes = asc.parameter_dict_get(parameters, 'stimFrames', 108000)
    preframes = asc.parameter_dict_get(parameters, 'preFrames', 200)
    nblinks = asc.parameter_dict_get(parameters, 'Nblinks', 2)

    seed = asc.parameter_dict_get(parameters, 'seed', -10000)
    seed2 = asc.parameter_dict_get(parameters, 'objseed', -1000)

    stepsize = asc.parameter_dict_get(parameters, 'stepsize', 2)

    ntotal = int(stimframes / nblinks)

    # Generate the numbers to be used for reconstructing the motion
    # ObjectsMovingBackground.cpp line 174, steps are generated in an alternating
    # fashion. We can generate all of the numbers at once (total lengths is
    # defined by stimFrames) and then assign to x and y directions.
    # Although there is more stuff around line 538
    randnrs, seed = randpy.gasdev(seed, ntotal*2)
    randnrs = np.array(randnrs)*stepsize

    xsteps = randnrs[::2]
    ysteps = randnrs[1::2]

    clusters, metadata = asc.read_spikesheet(exp_name)
    filter_length, frametimings = asc.ft_nblinks(exp_name, stimnr, nblinks,
                                                 metadata['refresh_rate'])
    clusterids = plf.clusters_to_ids(clusters)

    all_spikes = np.empty((clusters.shape[0], ntotal))
    for i, (cluster, channel, _) in enumerate(clusters):
        spiketimes = asc.read_raster(exp_name, stimnr, cluster, channel)
        spikes = asc.binspikes(spiketimes, frametimings)[:-1]
        all_spikes[i, :] = spikes

    # Collect STA for x and y movement in one array
    stas = np.zeros((clusters.shape[0], 2, filter_length))
    stc_x = np.zeros((clusters.shape[0], filter_length, filter_length))
    stc_y = np.zeros((clusters.shape[0], filter_length, filter_length))
    for k in range(filter_length, ntotal-filter_length+1):
        x_mini = xsteps[k-filter_length+1:k+1][::-1]
        y_mini = ysteps[k-filter_length+1:k+1][::-1]
        for i, (cluster, channel, _) in enumerate(clusters):
            if all_spikes[i, k] != 0:
                stas[i, 0, :] += all_spikes[i, k]*x_mini
                stas[i, 1, :] += all_spikes[i, k]*y_mini
                # Calculate non-centered STC (Cantrell et al., 2010)
                stc_x[i, :, :] += all_spikes[i, k]*calc_covar(x_mini)
                stc_y[i, :, :] += all_spikes[i, k]*calc_covar(y_mini)

    eigvals_x = np.zeros((clusters.shape[0], filter_length))
    eigvals_y = np.zeros((clusters.shape[0], filter_length))
    eigvecs_x = np.zeros((clusters.shape[0], filter_length, filter_length))
    eigvecs_y = np.zeros((clusters.shape[0], filter_length, filter_length))
    # Normalize STAs and STCs with respect to spike numbers
    for i in range(clusters.shape[0]):
        totalspikes = all_spikes.sum(axis=1)[i]
        stas[i, :, :] = stas[i, :, :] / totalspikes
        stc_x[i, :, :] = stc_x[i, :, :] / totalspikes
        stc_y[i, :, :] = stc_y[i, :, :] / totalspikes
        eigvals_x[i, :], eigvecs_x[i, :, :] = np.linalg.eigh(stc_x[i, :, :])
        eigvals_y[i, :], eigvecs_y[i, :, :] = np.linalg.eigh(stc_y[i, :, :])

    savepath = os.path.join(exp_dir, 'data_analysis', stimname)
    if not os.path.isdir(savepath):
        os.makedirs(savepath, exist_ok=True)

    for i in range(stas.shape[0]):
        stax = stas[i, 0, :]
        stay = stas[i, 1, :]
        ax1 = plt.subplot(211)
        ax1.plot(stax, label=r'$STA_{X}$')
        ax1.plot(stay, label=r'$STA_{Y}$')
        ax1.plot(eigvecs_x[i, :, -1], label='Eigenvector_X 0')
        ax1.plot(eigvecs_y[i, :, -1], label='Eigenvector_Y 0')
        plt.legend(fontsize='xx-small')

        ax2 = plt.subplot(425)
        ax3 = plt.subplot(427)
        ax2.set_yticks([])
        ax2.set_xticks([])
        ax3.set_yticks([])
        ax2.set_ylabel('Eigenvalues X')
        ax2.plot(eigvals_x[i, :], 'o', markerfacecolor='C0', markersize=4,
                 markeredgewidth=0)
        ax3.plot(eigvals_y[i, :], 'o', markerfacecolor='C1', markersize=4,
                 markeredgewidth=0)
        plf.spineless([ax1, ax2, ax3], 'tr')
        ax4 = plt.subplot(224, projection='polar')
        # Calculating based on the last eigenvector
        magx = eigvecs_x[i, :, -1].sum()
        magy = eigvecs_y[i, :, -1].sum()
        # Convert to polar coordinates using np.arctan2
        r = np.sqrt(magx**2 + magy**2)
        theta = np.arctan2(magy, magx)
        ax4.plot([0, theta], [0, r])
        plt.suptitle(f'{exp_name}\n{stimname}\n{clusterids[i]}')
        plt.savefig(os.path.join(savepath, clusterids[i]+'.svg'),
                    bbox_inches='tight')
        plt.close()

    # Calculated based on last eigenvector
    magx = eigvecs_x[:, :, -1].sum(axis=1)
    magy = eigvecs_y[:, :, -1].sum(axis=1)
    r_ = np.sqrt(magx**2 + magy**2)
    theta_ = np.arctan2(magy, magx)
    # Draw the vectors starting from origin
    r = np.zeros(r_.shape[0]*2)
    theta = np.zeros(theta_.shape[0]*2)
    r[::2] = r_
    theta[::2] = theta_
    plt.polar(theta, r)
    plt.title(f'Population plot for motion STAs\n{exp_name}')
    plt.savefig(os.path.join(savepath, 'population.svg'))
    plt.show()
    plt.close()

    keystosave = ['nblinks', 'all_spikes', 'clusters',
                  'eigvals_x', 'eigvals_y',
                  'eigvecs_x', 'eigvecs_y',
                  'filter_length', 'magx', 'magy',
                  'ntotal', 'r', 'theta', 'stas',
                  'stc_x', 'stc_y']
    datadict = {}

    for key in keystosave:
        datadict[key] = locals()[key]

    npzfpath = os.path.join(savepath, str(stimnr)+'_data')
    np.savez(npzfpath, **datadict)

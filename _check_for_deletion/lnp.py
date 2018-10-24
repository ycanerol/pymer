#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 22 14:58:43 2017

@author: ycan

Functions for LNP model with full field flicker stimulus

Includes STA, STC to be used in data analysis

"""
import numpy as np
from scipy.stats.mstats import mquantiles


def sta(spikes, stimulus, filter_length, total_frames):
    snippets = np.zeros(filter_length)
    for i in range(filter_length, total_frames):
        if spikes[i] != 0:
            snippets = snippets+stimulus[i:i-filter_length:-1]*spikes[i]
            # Snippets are inverted before being added
    sta_unscaled = snippets/np.sum(spikes)   # Normalize/scale the STA
    sta_scaled = sta_unscaled/np.sqrt(np.sum(np.power(sta_unscaled, 2)))
    return sta_scaled, sta_unscaled  # Unscaled might be needed for STC


def log_nlt_recovery(spikes, filtered_recovery, bin_nr, dt):
    logbins = np.logspace(0, np.log(30)/np.log(10), bin_nr)
    logbins = -logbins[::-1]+logbins
    logbindices = np.digitize(filtered_recovery, logbins)
    spikecount_in_logbins = np.array([])
    for i in range(bin_nr):
        spikecount_in_logbins = np.append(spikecount_in_logbins,
                                          (np.average(spikes[np.where
                                                             (logbindices == i)]))/dt)
    return logbins, spikecount_in_logbins


def q_nlt_recovery(spikes, filtered_recovery, bin_nr, dt):
    quantiles = mquantiles(filtered_recovery,
                           np.linspace(0, 1, bin_nr+1, endpoint=False)[1:])
    bindices = np.digitize(filtered_recovery, quantiles)
    # Returns which bin each should go
    spikecount_in_bins = np.array([])
    for i in range(bin_nr):  # Sorts values into bins
        spikecount_in_bins = np.append(spikecount_in_bins,
                                       (np.average(spikes[np.where
                                                          (bindices == i)])/dt))
    return quantiles, spikecount_in_bins


def stc(spikes, stimulus, filter_length, total_frames, dt,
        eigen_indices=[0, 1, -2, -1], bin_nr=60):
    # Gives non-centered STC
    covariance = np.zeros((filter_length, filter_length))
    sta_temp = sta(spikes, stimulus, filter_length, total_frames)[1]
    # Unscaled STA
    for i in range(filter_length, total_frames):
        if spikes[i] != 0:
            snippet = stimulus[i:i-filter_length:-1]
            # Snippets are inverted before being added
            # snpta = np.array(snippet-sta_temp)[np.newaxis, :] Centered STC
            snpta = np.array(snippet)[np.newaxis, :]  # Non-centered STC
            covariance = covariance+np.dot(snpta.T, snpta)*spikes[i]
    covariance = covariance/(sum(spikes)-1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)

    sorted_eig = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_eig]
    eigenvectors = eigenvectors[:, sorted_eig]

    # Calculating nonlinearities
    generator_stc = np.zeros((total_frames, len(eigen_indices)))
    bins_stc = np.zeros((bin_nr, len(eigen_indices)))
    spikecount_stc = np.zeros((bin_nr, len(eigen_indices)))
    eigen_legends = []

    for i in range(len(eigen_indices)):
        generator_stc[:, i] = np.convolve(eigenvectors[:, eigen_indices[i]],
                                          stimulus,
                                          mode='full')[:-filter_length+1]
        bins_stc[:, i],\
        spikecount_stc[:, i] = q_nlt_recovery(spikes,
                                              generator_stc[:, i], 60, dt)
        if eigen_indices[i] < 0:
            eigen_legends.append('Eigenvector {}'
                                 .format(filter_length+int(eigen_indices[i])))
        else:
            eigen_legends.append('Eigenvector {}'.format(int(eigen_indices[i])))

    return eigenvalues, eigenvectors, bins_stc, spikecount_stc, eigen_legends

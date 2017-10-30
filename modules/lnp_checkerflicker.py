#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 24 16:41:38 2017

@author: ycan

Functions for STA and STC analysis for checkerflicker stimulus

"""

import numpy as np
from scipy.stats.mstats import mquantiles
import scipy.ndimage as ndi
import peakutils


def sta(spikes, stimulus, filter_length, total_frames):
    sx = stimulus.shape[0]
    sy = stimulus.shape[1]
    snippets = np.zeros((sx, sy, filter_length))
    for i in range(filter_length, total_frames-filter_length+1):
        if spikes[i] != 0:
            stimulus_reversed = stimulus[:, :, i-filter_length+1:i+1][:,:,::-1]
            snippets = snippets+stimulus_reversed*spikes[i]
            # Snippets are inverted before being added
    sta_unscaled = snippets/np.sum(spikes)   # Normalize/scale the STA

    sta_gaussian = ndi.filters.gaussian_filter(sta_unscaled, sigma=(1, 1, 0))
    # Gaussian is applied before searching for the brightest/darkest pixel
    # to exclude randomly high values outside the receptive field
    max_i = np.squeeze(np.where(np.abs(sta_gaussian) ==
                                np.max(np.abs(sta_gaussian))))
    temporal = sta_unscaled[max_i[0], max_i[1], :].reshape(20,)
    temporal = temporal / np.sqrt(np.sum(np.power(temporal, 2)))

    return sta_unscaled, max_i, temporal
    # Unscaled might be needed for STC


def check_max_i(sta, max_i):
    # Checks if max_i is too close to the borders
    # Resets is to the closest possible if so
    f_size = 5

    sx = sta.shape[0]
    sy = sta.shape[1]

    x_d1 = max_i[0]-f_size
    x_d2 = max_i[0]+f_size
    y_d1 = max_i[1]-f_size
    y_d2 = max_i[1]+f_size

    if x_d1 <= 0: max_i[0] = f_size+1
    elif x_d2 > sx-1: max_i[0] = sx-f_size-1

    if y_d1 <= 0: max_i[1] = f_size+1
    elif y_d2 > sy-1: max_i[1] = sy-f_size-1
    return max_i


def stim_weighted(sta, max_i, stimulus):
    # Turns the checkerflicker stimulus into more Gaussian-like
    f_size = 5

    weights = sta[max_i[0]-f_size-1:max_i[0]+f_size,
                  max_i[1]-f_size-1:max_i[1]+f_size,
                  max_i[2]].reshape((2*f_size+1, 2*f_size+1))
    if weights.max() < np.max(np.abs(weights)):
        weights = -weights
    weights = weights/np.sqrt(np.sum(weights**2))
    stim_small = stimulus[max_i[0]-f_size-1:max_i[0]+f_size,
                          max_i[1]-f_size-1:max_i[1]+f_size, :]
    stim_weighted = np.array([])
    for i in range(stim_small.shape[2]):
        stim_weighted = np.append(stim_weighted, np.sum(stim_small[:, :, i] *
                                                        weights))
    return stim_weighted


def nlt_recovery(spikes, stimulus, sta, bin_nr, dt):
    generator = np.convolve(sta, stimulus, mode='full')[:-sta.size+1]
    quantiles = mquantiles(generator,
                           np.linspace(0, 1, bin_nr+1, endpoint=False)[1:])
    bindices = np.digitize(generator, quantiles)
    # Returns which bin each should go
    spikecount_in_bins = np.array([])
    for i in range(bin_nr):  # Sorts values into bins
        spikecount_in_bins = np.append(spikecount_in_bins,
                                       (np.average(spikes[np.where
                                                          (bindices == i)])/dt))
    return quantiles, spikecount_in_bins


def stc(spikes, stimulus, filter_length, total_frames, dt,
        eigen_indices=[0, 1, -2, -1], bin_nr=60):
    # Non-centered STC
    covariance = np.zeros((filter_length, filter_length))
    for i in range(filter_length, total_frames):
        if spikes[i] != 0:
            snippet = stimulus[i:i-filter_length:-1]
            snippet = np.reshape(snippet, (1, 20))
            covariance = covariance+np.dot(snippet.T, snippet)*spikes[i]
    covariance = covariance/(np.sum(spikes)-1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)

    sorted_eig = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_eig]
    eigenvectors = eigenvectors[:, sorted_eig]

    # Calculating nonlinearities
    generator_stc = np.zeros((total_frames, len(eigen_indices)))
    bins_stc = np.zeros((bin_nr, len(eigen_indices)))
    spikecount_stc = np.zeros((bin_nr, len(eigen_indices)))
    eigen_legends = []

    return eigenvalues, eigenvectors, bins_stc, spikecount_stc, eigen_legends


def getpeak(temporal_filter):
    threshold = .7
    peaks = peakutils.indexes(temporal_filter, thres=threshold)
    peaks = np.append(peaks, peakutils.indexes(-temporal_filter,
                                               thres=threshold))
    sorted_peaks = np.argsort(np.abs(temporal_filter[peaks]))[::-1]
    peaks = peaks[sorted_peaks][:2]  # Get two largest peaks
    peak = peaks.min()  # Get the earliest peak
    return peak


def onoffindex(temporal_filter, bins, spikecount_in_bins):
    # Get peaks of STA/STC
#    peak = np.argmax(np.abs(temporal_filter[:7]))
    peak = getpeak(temporal_filter)
    # Flip if positive
    if temporal_filter[peak] < 0:
        temporal_filter = -temporal_filter
        spikecount_in_bins = spikecount_in_bins[::-1]
    # integrate non-linearity bin size * fire rate
    on_ind = np.where(bins > 0)
    off_ind = np.where(bins <= 0)
    r_on = np.trapz(spikecount_in_bins[on_ind], bins[on_ind])
    r_off = np.trapz(spikecount_in_bins[off_ind], bins[off_ind])
    onoffindex = (r_on-r_off)/(r_on+r_off)
    return temporal_filter, bins, spikecount_in_bins, peak, onoffindex

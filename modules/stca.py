#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""

import datetime as dt

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats.mstats import mquantiles

import analysis_scripts as asc

from omb import OMB


def calc_stca(spikes, stimulus, filter_length):
    rw = asc.rolling_window(stimulus, filter_length, preserve_dim=True)
    sta, stc = calc_stca_from_stimulus_matrix(spikes, rw)
    return sta, stc


def calc_stca_from_stimulus_matrix(spikes, stimulus_matrix, sta=None):
    """
    Use the stimulus matrix generated by rolling window to calculate STA and STC.

    This allows for reusing the stimulus matrix with some components projected out.

    When calculating STC, STA is subtracted from the stimulus but not projected out.

    stimulus_matrix is equivalent to the output of the rolling window function.
    It is a (N, f) matrix where each row contains the preceding stimulus
    segment. Total number of time bins is N and the lenght of the filter in
    number of bins is f.

    When using this function with modified stimulus matrices, original STA
    can be supplied so that is computed on the original stimulus distribution.
    """
    if sta is None:
        sta = (spikes @ stimulus_matrix) / spikes.sum()
    precovar = (stimulus_matrix * spikes[:, None]) - sta
    stc = (precovar.T @ precovar) / (spikes.sum()-1)
    return sta, stc


def project_component_out_stimulus_matrix(stimulus_matrix, componentstoremove):
    """
    Project given components out of the stimulus matrix.

    componentstoremove:
        Each row should be one component to project out of the stimulus
    """
    ncomp = componentstoremove.shape[0]
    reduced_stim_matrix = stimulus_matrix.copy()
    for i in range(ncomp):
        comp = componentstoremove[i, :]
        compdotcomp = np.dot(comp, comp)

        term1 = np.einsum('ij,j->i', reduced_stim_matrix, comp) / compdotcomp
        term2 = np.einsum('i,j->ij', term1, comp)
        reduced_stim_matrix = reduced_stim_matrix - term2
        # Slow old version.
#        for j in range(reduced_stim_matrix.shape[0]):
#            reduced_stim_matrix[j, :] = (reduced_stim_matrix[j, :]
#                - np.dot(np.dot(reduced_stim_matrix[j, :], comp) / compdotcomp, comp))
    return reduced_stim_matrix


def confidence_interval_bootstrap(data, confidence_level=.95):
    """
    Calculate a confidence interval for a 1-D dataset using bootstrapped values.

    """
    a = (1 - confidence_level)/2
    low, high = mquantiles(data, [a, 1-a])
    return low, high


def sigtest(spikes, stimulus, filter_length, ntest=500,
            confidence_level=.95, debug_plot=False):
    """
    Calculate the significant components of the spike-triggered covariance
    matrix.

    Parameters
    ---------


    Returns
    -------
    significant_components:
        indices of the significant components of the spike-triggered covariance
        matrix, matching the return value of np.linalg.eigh(stc).
        Corresponds to the nth element of the eigenvalues and nth column
        of the eigenvectors.

    Example
    -------
    >>> sta, stc = calc_stca(spikes, stimulus, 20)
    >>> eigenvalues, eigenvectors = np.linalg.eigh(stc)
    >>> sig_comp_inds = sigtest(spikes, stimulus, 20)
    >>> print(sig_comp_inds)
    [0, 19]
    >>> significant_components = eigenvalues[:, sig_comp_inds].T

    """
    sta_init, stc_init = calc_stca(spikes, stimulus, filter_length)
    eigvals_init, eigvecs_init = np.linalg.eigh(stc_init)
    no_sig_comp_left = False
    significant_components = np.array([], dtype=np.int)
    # Keep track of components above and below the mean value
    # of eigenvectors to calculate the indices correctly
    ncomps_above, ncomps_below = 0, 0
    while not no_sig_comp_left:
        all_v = np.zeros((2, ntest))  # first axis: min, max eigvals

        toremove = eigvecs_init[:, significant_components].T
        rw = asc.rolling_window(stimulus, filter_length, preserve_dim=True)
        reduced_stim_matrix = project_component_out_stimulus_matrix(rw,
                                                                    toremove)
        _, stc_loop = calc_stca_from_stimulus_matrix(spikes,
                                                     reduced_stim_matrix,
                                                     sta=sta_init)

        eigvals_loop, _ = np.linalg.eigh(stc_loop)
        eigvals_notzero = eigvals_loop[~np.isclose(eigvals_loop, 0, atol=1e-2)]

        for i in range(ntest):
            shifted_spikes = np.roll(spikes, np.random.randint(spikes.shape[0]))
            _, r_stc = calc_stca_from_stimulus_matrix(shifted_spikes,
                                                      reduced_stim_matrix,
#                                                      sta=sta_init
                                                      )

            rand_v, _ = np.linalg.eigh(r_stc)
            # Exclude the zero eigenvalues corresponding to the significant components
            rand_v = np.ma.masked_values(rand_v, value=0, atol=1e-2)
            all_v[:, i] = np.array([rand_v.min(), rand_v.max()])

        low_min, high_min = confidence_interval_bootstrap(all_v[0],
                                                          confidence_level)
        low_max, high_max = confidence_interval_bootstrap(all_v[1],
                                                          confidence_level)

        outliers_low = np.where(eigvals_notzero < low_min)[0]
        outliers_high = np.where(eigvals_notzero > high_max)[0]

        outlier_inds = np.hstack((outliers_low, outliers_high))
        outliers = eigvals_notzero[outlier_inds]
        if len(outliers_low) + len(outliers_high) == 0:
            no_sig_comp_left = True
        else:
            dist_low_to_min = low_min - eigvals_notzero[outliers_low]
            dist_high_to_max = eigvals_notzero[outliers_high] - high_max
            dist_to_extrema = np.hstack((dist_low_to_min, dist_high_to_max))
            largest_outlier_ind = np.argmax(dist_to_extrema)

            largest_outlier = outliers[largest_outlier_ind]
            outlier_index = np.where(eigvals_notzero == largest_outlier)[0]

            # Each time a new component from below the line is added, the returned
            # index decreases by one. We correct for this.
            outlier_index = outlier_index + ncomps_below
            significant_components = np.hstack((significant_components, outlier_index))

            if outlier_index == len(eigvals_notzero) - 1:
                # Outlier larger than mean eigenvalue
                ncomps_above += 1
            elif outlier_index == 0 + ncomps_below:
                # Outlier smaller than mean eigenvalue
                ncomps_below += 1
            else:
                raise ValueError('Largest outlier found in unexpected place!')

            if debug_plot:
                plt.figure()
                plt.plot(eigvals_notzero, 'ko')
                for line in [low_min, high_min, low_max, high_max]:
                    plt.axhline(line, color='red', alpha=.3)

        if len(significant_components) > 8:
            raise ValueError('Number of significant components is too damn high!')
    return significant_components

#%%
if __name__ == '__main__':

    exp, stimnr = '20180710', 8

    st = OMB(exp, stimnr)
    stimulus = st.bgsteps[0, :]
    allspikes = st.allspikes()
    filter_length = st.filter_length


    from stimulus import Stimulus
    from randpy import randpy
    ff = Stimulus('20180710', 1)
    stimulus = np.array(randpy.gasdev(-1000, ff.frametimings.shape[0])[0])
    filter_length = ff.filter_length
    allspikes = ff.allspikes()
    st = ff

    rw = asc.rolling_window(stimulus, filter_length, preserve_dim=True)

    start_time = dt.datetime.now()
    # Calculate the significant components of STC for all cells
    sig_comps = []
    for i in range(st.nclusters):
        inds = sigtest(allspikes[i, :], stimulus, filter_length)
        sig_comps.append(inds)

    # Plot the significant components
    for i in range(st.nclusters):
        sig_comp = sig_comps[i]
        sta, stc = calc_stca(allspikes[i, :], stimulus, filter_length)
        eigvals, eigvecs = np.linalg.eigh(stc)
        fig, axes = plt.subplots(2, 2)
        axes = axes.flat
        axes[0].imshow(stc)
        axes[1].plot(eigvals, 'ko')
        for comp in sig_comp:
            axes[1].plot(comp, eigvals[comp], 'o')
            axes[2].plot(eigvecs[:, comp])
        axes[2].plot(sta, 'k')
        plt.suptitle(f'cell {i}, {st.clids[i]}, nsp={allspikes[i, :].sum():6.0f}')
        plt.show()

    elapsed = dt.datetime.now() - start_time
    print(f'Took {elapsed.total_seconds()/60:4.2f} minutes'
          f' for {st.nclusters} cells.')
import numpy as np
from scipy.linalg import fractional_matrix_power

import spikeshuffler

def cca_macke(spikes, stimulus, filter_length, n_components):
    """
    Calculates CCA components 'manually' i.e. matrix multiplications
    and SVD. This method is similar to Kuehn et al. 2019 and
    Macke et al. 2008.

    Parameters:
    -----
        spikes:
            spikes in the experiment shaped in a rolling-window fashion with
            cells flattened.
            Should have the shape N x (ncells x filter_length) where N is the
            number of total frames in the experiment.
        stimulus:
            The stimulus shaped in the same way spikes are. Instead of ncells,
            number of stimulus dimensions are used i.e N x (stimdim x filter_length)

        filter_length:
            Number of frames of stimulus and spikes that were used when reshaping
            the spikes and stimulus matrices.
        n_components:
            Number of components that are requested. All of them are calculated
            but only the first n_components are returned.

    Returns:
    -----
        response:

        stim:

        can_correlations:
           Cannonical correlation values.
    """

    ncells = int(spikes.shape[1]/filter_length[1])

    stimavg = stimulus.mean(axis=0)[np.newaxis, :]
    spikesavg = spikes.mean(axis=0)[np.newaxis, :]

    #%%
    stimcov = np.cov(stimulus.T)
    spkcov = np.cov(spikes.T)

    stspcov = np.zeros((stimcov.shape[0], spkcov.shape[0]))

    for i in range(stspcov.shape[0]):
        for j in range(stspcov.shape[1]):
            stspcov[i, j] = np.mean(
                    (stimulus[:, i] - np.mean(stimulus[:, i]))*
                    (spikes[:, j] - np.mean(spikes[:, j]))
                                    )
    #%%
    ncomponents = min(stimulus.shape[1], spikes.shape[1])

    stimcov_exp = fractional_matrix_power(stimcov, -0.5)
    spkcov_exp = fractional_matrix_power(spkcov, -0.5)
    whitened_cov =  stimcov_exp @ stspcov @ spkcov_exp

    u, s, v = np.linalg.svd(whitened_cov, full_matrices=True)

    cancorrs = s
    # rows of v correspond to components, to interpret it in the same way,
    # we transpose it
    v = v.T
    print('done')

    # Back-transform the matrix after whitening.
    u = stimcov_exp @ u
    v = spkcov_exp @ v

    # SVD returns a much larger matrix than needed for the larger matrix, we take
    # only the needed part.
    resp_comps = v[:, :ncomponents].reshape((ncells, filter_length[1], ncomponents))
    # Reshape responses so that the order is: ncomponents, ncells, time
    resp_comps = np.moveaxis(resp_comps, [2, 0, 1], [0, 1, 2])

    stim_comps = u.reshape((2, filter_length[0], ncomponents))
    # Reshape stimulus components so that the order is: ncomponents, ndims, time
    stim_comps = np.moveaxis(stim_comps, [2, 0, 1], [0, 1, 2])

    return resp_comps, stim_comps, cancorrs


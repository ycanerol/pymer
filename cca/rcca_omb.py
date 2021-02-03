import numpy as np
import rcca


whiten = True

def whiten_data(data: np.array):
    """
    Whiten any given data and return the matrix that will transform
    the data into original coordinates.

    Rows of the input data matrix correspond to the individual dimensions,
    columns are the features.

    Original source:
    https://theclevermachine.wordpress.com/2013/03/30/the-statistical-whitening-transform/

    Example
    ----
    whitened, rotation = whiten_data(data)

    [do your anaylsis on whitened data here]
    result = get_components(whitened)

    results_unrotated = result @ rotation
    """
    if not np.isclose(np.mean(data), 0, atol=1e-2):
        raise ValueError('Data should be mean-subtracted before whitening.')

    # Decorrelate the data by calculating the
    # eigenvectors of its covariance matrix
    # and multiplying the matrix by it
    #
    # Technically, we multiply by transpose of this matrix
    # but they are equal (as well as the inverse of this matrix)
    from datetime import datetime
    from miscfuncs import timediff

    # We want the to apply the whitening between different cells, not across time
    data = data.T

    start = datetime.now()
    print(f'{timediff(start)}  Calculating covariance matrix with {data.shape=}')
    covx = np.cov(data)
    print(f'{timediff(start)}  Covariance calculated, starting eigendecomposition ')
    eigvals, eigvecs = np.linalg.eigh(covx)
    rotation = eigvecs
    print(f'{timediff(start)}  Eigenvector calculation complete {eigvals.shape=} {eigvecs.shape=}')
    print(f'{timediff(start)}  Rotating the data matrix')
    data_decor = rotation @ data
    print(f'{timediff(start)}  Data matrix rotated')

    # We scale the matrix by multiplying it by D^(-1/2)
    print(f'{timediff(start)}  Calculating D^(-1/2)')
    # Make sure the eigenvalues are not zero
    eigvals += 1e-10
    D = np.diag(eigvals ** (-0.5))

    print(f'{timediff(start)}  D calculated. Scaling the data.')
    data_decor_scaled = D @ data_decor
    print(f'{timediff(start)}  Whitening completed.')

    data_decor_scaled_transposed = data_decor_scaled.T
    # The resulting whitened matrix is in rotated coordinates
    # The rotation can be reversed by multiplying by the inverse
    # of the rotation matrix, which is equal to the rotation
    # matrix itself.
    return data_decor_scaled_transposed, rotation


def cca_rcca(spikes, stimulus, filter_length, n_components,
             regularization, whiten):

    ncells = int(spikes.shape[1]/filter_length[1])
    if whiten:
        spikes, spikes_rotation = whiten_data(spikes)

    cca = rcca.CCA(kernelcca=False, reg=regularization, numCC=n_components)
    cca.train([spikes, stimulus])

    spikes_res = cca.ws[0]
    # Derotate the spikes to be able to interpret the responses
    if whiten:
        spikes_res = spikes_rotation @ spikes_res

    resp_comps = np.swapaxes(spikes_res, 1, 0)
    resp_comps = resp_comps.reshape((n_components, ncells, filter_length[1]))

    stim_comps = np.swapaxes(cca.ws[1], 1, 0)
    stim_comps = stim_comps.reshape((n_components, 2, filter_length[0]))

    cancorrs = cca.cancorrs

    return resp_comps, stim_comps, cancorrs


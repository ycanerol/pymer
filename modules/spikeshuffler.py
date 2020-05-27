import numpy as np


def randomize_spikes(spikes):
    """
    for binned spikes
    """
    return np.random.permutation(spikes)


def randomize_spiketimes(spiketimes):
    """
    for spike times (not binned)
    """
    isi = np.ediff1d(spiketimes)
    shuffled_isi = np.random.permutation(isi)
    shuffled_spiketimes = np.cumsum(shuffled_isi)

    return shuffled_spikes


def shufflecolumns(X):
    """
    Shuffle the columns of a given array.

    Notes
    ------

    Note that this function preserves the structure of columns i.e. the
    simultaneous spikes from different cells.

    >>> x = np.arange(20).reshape(2, 10)
    >>> shufflecolumns(x)
    array([[ 5,  9,  3,  7,  6,  1,  0,  2,  8,  4],
           [15, 19, 13, 17, 16, 11, 10, 12, 18, 14]])

    Adapted from https://stackoverflow.com/a/35647011/9205838
    """
    return np.take(X, np.random.permutation(X.shape[1]),axis=1)


def shufflebyrow(X):
    """
    Shuffle each row of a given array independently. Can be useful for
    randomizing binned spikes from all cells, for statistical testing.

    Parameters
    --------

    X:  np.array with two dimensions
        Rows should correspond to cells, columns to time bins.

    Example
    ------

    >>> allspikes.shape
    (37, 10000)
    >>> shuffledspikes = shufflebyrow(allspikes)
    >>> np.allclose(allspikes.sum(axis=1),
                    shuffledspikes.sum(axis=1))
    True

    Notes
    -----
    Adapted from https://github.com/numpy/numpy/issues/5173#issuecomment-467625388
    """
    if X.ndim > 2:
        raise ValueError('Maximum 2 dimensions are allowed in the spike array.')
    return np.apply_along_axis(np.random.permutation, axis=1, arr=X)


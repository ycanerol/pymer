
import numpy as np
import analysis_scripts as asc

def train_test_split(spikes, stimulus, test_size=0.2, split_pos=None):
    """
    split_pos:
        Where the split should start as a ratio.

    Returns:
        spikes_training, spikes_test, stimulus_training, stimulus_test
    """
    total_len = spikes.shape[0]
    if split_pos is None:
        split_pos = np.random.rand()*(1-test_size)
    split_ind = int(split_pos * total_len)
    split_end = split_ind + int(test_size * total_len)
    mask = np.array([True]*total_len)
    mask[split_ind:split_end] = False
    spikes_training = spikes[mask]
    spikes_test = spikes[~mask]
    stimulus_training = stimulus[..., mask]
    stimulus_test = stimulus[..., ~mask]
    return spikes_training, spikes_test, stimulus_training, stimulus_test


def packdims(array, window):
    sh = array.shape
    if array.ndim == 1:
        array = array[None, :]
    if array.ndim > 2:
        array = array.reshape(np.prod(sh[:-1]), -1)
    rw = np.empty((sh[-1], array.shape[0]*window))
    for i in range(array.shape[0]):
        rw[:, i*window:(i+1)*window] = asc.rolling_window(array[i, :], window)
    return rw


def shiftspikes(spikes: np.array, shift:int) -> np.array:
    """
    Shift the spikes array by a given amount by padding with zeros
    at the beginning and trimming off the end.

    """
    shift = int(shift)
    if shift == 0:
        return spikes
    return np.hstack((np.zeros((spikes.shape[0],
                                shift)), spikes))[:, :-shift]


def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""

from matplotlib import pyplot as plt
import numpy as np

from scipy.stats.mstats import mquantiles
from scipy.stats import binned_statistic, binned_statistic_2d


def bin_midpoints(bins):
    """
    Calculate the middle point of the bins

    This will reduce the size of the array by one.
    """
    return (bins[1:]+bins[:-1])/2


def _calc_nonlin(spikes, generator, nr_bins=20):
    """
    This is the previous version of the function, without using
    binned statistic. It is easier to see the logic, so this is
    left for posterity.

    NOTE:
    There is a slight difference between the old and new versions;
    old version has slightly fewer count in the last bin.

    Calculate nonlinearities from the spikes and the generator signal.
    Bins for the generator are defined such that they contain equal number
    of samples. Since there are fewer samples for more extreme values of the
    generator signal, bins get wider.
    """

    quantiles = np.linspace(0, 1, nr_bins+1)

    quantile_bins = mquantiles(generator, prob=quantiles)
    bindices = np.digitize(generator, quantile_bins)
    # Returns which bin each should go
    spikecount_in_bins = np.full(nr_bins, np.nan)
    for i in range(nr_bins):  # Sorts values into bins
        spikecount_in_bins[i] = spikes[bindices == i+1].mean()
    # Use the middle point of adjacent bins instead of the edges
    # Note that this decrasese the length of the array by one
    quantile_bins = (quantile_bins[1:]+quantile_bins[:-1])/2
    return quantile_bins, spikecount_in_bins


def calc_nonlin(spikes, generator, nr_bins=20):
    """
    Calculate nonlinearities from the spikes and the generator signal.
    Bins for the generator are defined such that they contain equal number
    of samples. Since there are fewer samples for more extreme values of the
    generator signal, bins get wider.
    """

    quantiles = np.linspace(0, 1, nr_bins+1)

    # m stands for masked, to be able to apply the function
    # to masked numpy arrays. In practice, masked arrays are rarely needed.
    quantile_bins = mquantiles(generator, prob=quantiles)

    res = binned_statistic(generator, spikes, bins=quantile_bins)

    nonlinearity = res.statistic
    bins = bin_midpoints(quantile_bins)

    return nonlinearity, bins


#%%
def calc_nonlin_2d(spikes, generator_x, generator_y, nr_bins=20):

    quantiles = np.linspace(0, 1, nr_bins+1)
    qbins_x = mquantiles(generator_x, prob=quantiles)
    qbins_y = mquantiles(generator_y, prob=quantiles)

    res = binned_statistic_2d(generator_x,
                              generator_y,
                              spikes,
                              bins=[qbins_x, qbins_y])

    nlt_2d = res[0]
    bins_x = bin_midpoints(qbins_x)
    bins_y = bin_midpoints(qbins_y)

    return nlt_2d, bins_x, bins_y


if __name__ == '__main__':

    from omb import OMB

    exp, stim = '20180710', 8

    st = OMB(exp, stim)

    data = st.read_datafile()

    generators_x = data['generators_x']
    generators_y = data['generators_y']

    i = 0
    spikes = st.allspikes()

    nlt, bins_x, bins_y = calc_nonlin_2d(spikes[i, :],
                                         generators_x[i, :],
                                         generators_y[i, :],
                                         nr_bins=9)

    # Normalize nonlinearity so units are spikes/s
    nlt = nlt/st.frame_duration

    cmap = 'Greens'
    cmap_contours = 'coolwarm'
    #%%
    plt.figure()
    plt.imshow(nlt, cmap=cmap)
    plt.show()

    X, Y = np.meshgrid(bins_x, bins_y)

    plt.figure()
    plt.pcolormesh(X, Y, nlt, cmap=cmap)
#    plt.contour(X, Y, nlt, 2)
    plt.axis('equal')
    plt.show()

    #%%
    from mpl_toolkits import mplot3d
    fig = plt.figure()
    ax = plt.axes(projection='3d',)
    ax.plot_surface(X, Y, nlt, cmap=cmap)
    ax.set_xlabel(r'Stimulus projection $\bf{X}$')
    ax.set_ylabel(r'Stimulus projection $\bf{Y}$')
    ax.set_zlabel(r'Firing rate [Hz]')

    # Set the view so that maximum edges of X and Y axes are closest
    ax.view_init(elev=30, azim=45)

    # Where to plot the contours
    xoffset = min(ax.get_xlim())
    yoffset = min(ax.get_ylim())

    ax.contour(X, Y, nlt, zdir='y', offset=xoffset, cmap=cmap_contours)
    ax.contour(X, Y, nlt, zdir='x', offset=yoffset, cmap=cmap_contours)
    ax.contour(X, Y, nlt, 3, zdir='z', offset=nlt.min(), cmap=cmap)

    plt.suptitle(f'{exp} cell {i}')
    plt.show()

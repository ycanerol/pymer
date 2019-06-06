#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""

from matplotlib import pyplot as plt
import numpy as np

from scipy.stats.mstats import mquantiles
from scipy.stats import binned_statistic_2d


def bin_midpoints(bins):
    """
    Calculate the middle point of the bins

    This will reduce the size of the array by one.
    """
    return (bins[1:]+bins[:-1])/2


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
    import genlinmod as glm

    import iofuncs as iof
    exp, stim = '20180710', 8
    data = iof.load(exp, stim)

    stim = glm.loadstim(exp, stim, maxframenr=None)

    spikes = data['all_spikes']
    frame_duration = data['frame_duration']

    generators_x = data['generators_x']
    generators_y = data['generators_y']

    i = 0

    nlt, bins_x, bins_y = calc_nonlin_2d(spikes[i, :],
                                         generators_x[i, :],
                                         generators_y[i, :],
                                         nr_bins=9)

    # Normalize nonlinearity so units are spikes/s
    nlt = nlt/frame_duration

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

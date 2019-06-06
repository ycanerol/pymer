#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#%%

from matplotlib import pyplot as plt
import numpy as np

from scipy.stats.mstats import mquantiles
from scipy.stats import binned_statistic

def nlt_scpy(spikes, generator, nr_bins=20, **kwargs):
    quantiles = np.linspace(0, 1, nr_bins+1)
    qbins = mquantiles(generator, prob=quantiles)

    res = binned_statistic(generator, spikes, bins=qbins, **kwargs)
    nlt = res[0]
    bins = res[1]

    # Use the middle point for all bins
    bins = (bins[1:]+bins[:-1])/2

    return bins, nlt


if __name__ == '__main__':

    import iofuncs as iof
    import genlinmod as glm

    data = iof.load('20180710', 1)

    spikes = data['all_spiketimes']
    stim  = glm.loadstim('20180710', 1)

    i = 0
    sta = data['stas'][i]

    generator = np.convolve(stim, sta)[:-40+1]

    bins, nlt = nlt_scpy(spikes[i, :], generator, nr_bins=50)
    nlt_std = nlt_scpy(spikes[i, :], generator, nr_bins=50,
                       statistic='std')[1]

    plt.plot(bins, nlt, '.--')
    plt.fill_between(bins, nlt-nlt_std**2, nlt+nlt_std**2, alpha=0.2)
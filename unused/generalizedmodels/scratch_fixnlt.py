#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""

def q_nlt_recovery(spikes, generator, nr_bins=20):
    """
    Calculate nonlinearities from
    """
    # Define the quantiles we want to use for binning.
    # endpoint and [1:] are used to exclude outermost bins because
    # sometimes they cause bugs (e.g. nonlinearity is zero for some cells)

#    quantiles = np.linspace(0, 1, nr_bins+1, endpoint=False)[1:]
    quantiles = np.linspace(0, 1, nr_bins)

    quantile_bins = mquantiles(generator, prob=quantiles)
    bindices = np.digitize(generator, quantile_bins)
    # Returns which bin each should go
    spikecount_in_bins = np.array([])
    for i in range(nr_bins):  # Sorts values into bins
        spikecount_in_bins = np.append(spikecount_in_bins,
                                       (np.average(spikes[np.where
                                                          (bindices == i)])))
    return quantile_bins, spikecount_in_bins


#%%
def q_nlt_recovery2(spikes, generator, nr_bins=20):
    """
    Calculate nonlinearities from
    """
    # Define the quantiles we want to use for binning.
    # endpoint and [1:] are used to exclude outermost bins because
    # sometimes they cause bugs (e.g. nonlinearity is zero for some cells)

#    quantiles = np.linspace(0, 1, nr_bins+1, endpoint=False)[1:]
    quantiles = np.linspace(0, 1, nr_bins+1)

    quantile_bins = mquantiles(generator, prob=quantiles)
    bindices = np.digitize(generator, quantile_bins)
    plt.plot(np.bincount(bindices), 'o')
    plt.show()
    # Returns which bin each should go
    spikecount_in_bins = np.full(nr_bins, np.nan)
    for i in range(nr_bins):  # Sorts values into bins
        spikecount_in_bins[i] = spikes[bindices == i+1].mean()
    quantile_bins = (quantile_bins[1:]+quantile_bins[:-1])/2
    return quantile_bins, spikecount_in_bins

bins2, spikecount2 = q_nlt_recovery2(spikes, generator)
plt.plot(bins2, spikecount2/bin_length)

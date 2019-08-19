#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""

import numpy as np
import matplotlib.pyplot as plt


#%%  Simulation
dt = 0.016
filter_length = 20
stim = np.random.normal(size=5000)


t = np.arange(0, filter_length*dt, dt)

linear_filter = np.zeros(filter_length)
linear_filter[13:17] = [.2, .7, .1, -.5]


nonlinearity = lambda x: np.exp(x)/8
#nonlinearity = lambda x: np.maximum(x*3, 0)

# HINT: Linear filter is inverted here, alternative is flipping the
# stimulus snippets in the STA loop. Note the same inversion has to
# be repeated when calculating the generator signal using STA.

generator = np.convolve(stim, linear_filter[::-1], 'full')[:-filter_length+1]
firing_rate = nonlinearity(generator)

spikes = np.random.poisson(firing_rate)

#%%  Recovery
sta = np.zeros(filter_length)

for i, spike in enumerate(spikes):
    if i >= filter_length:
        sta += stim[i-filter_length+1:i+1]*spike
sta /= spikes.sum()

plt.plot(linear_filter, label='Filter')
plt.plot(sta, label='Spike-triggered average')
plt.legend(fontsize='x-small')
plt.show()
#%%
import nonlinearity as nlt

regenerator = np.convolve(sta[::-1], stim, 'full')[:-filter_length+1]

nonlin, bins = nlt.calc_nonlin(spikes, regenerator)
binsold, nonlinold = nlt._calc_nonlin(spikes, regenerator)

plt.plot(bins, nonlin, label='new')
plt.plot(binsold, nonlinold, label='old')
plt.plot(bins, nonlinearity(bins), label='real_nonlinearity')
plt.legend()

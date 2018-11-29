#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import plotfuncs as plf

def conv(k, x):
    return np.convolve(k, x, 'full')[:-k.shape[0]+1]

def glm_fr(k, mu):
    return lambda x:np.exp((conv(k, x) + mu)) # exponential
#    return lambda x:3/(1+np.exp(-(conv(k, x) + mu))) # logistic

def minimize_loglhd(k_initial, mu_initial, x, time_res, spikes):

    def loglhd(kmu):
        k_ = kmu[:-1]
        mu_ = kmu[-1]
        nlt_in = (conv(k_, x)+mu_)
        return -np.sum(spikes * nlt_in) + time_res*np.sum(np.exp(nlt_in))

    res = minimize(loglhd, [*k_initial, mu_initial], tol=1e-2)
    return res

#%%
filter_length = 40
frame_rate = 60
time_res = (1/frame_rate)
tstop = 5 # in seconds
t = np.arange(0, tstop, time_res)
np.random.seed(1221)

# Initialize model neuron
k_real = np.exp(-(t[:filter_length]-0.12)**2/.002)
k_real = np.exp(-(t[:filter_length]-.14)**2/.002)-np.exp(-(t[:filter_length]-.17)**2/.001)
mu_real = .1
f = glm_fr(k_real, mu_real)

# Generate stimulus
x = np.random.normal(size=t.shape)

# Calculate the firing rate and spikes of the neuron given the stimulus
rate = f(x)
spikes = np.random.poisson(rate)

np.random.seed()
k_guess = np.random.sample(size=filter_length)-.5
mu_guess = spikes.mean()*time_res


res = minimize_loglhd(k_guess, mu_guess, x, time_res, spikes)
k_res, mu_res = res['x'][:-1], res['x'][-1]

#%%
fig, axes = plt.subplots(3, 1, sharex=True)
[axgen, axfr, axsp] = axes.ravel()
axgen.plot(t, conv(k_real, x))
axgen.plot(t, conv(k_res, x), lw=.7)
axgen.set_ylabel('Generator')
axfr.plot(t, rate)
axfr.plot(t, glm_fr(k_res, mu_res-4)(x), lw=.8)
axfr.set_ylabel('Firing rate')
axsp.bar(t, spikes, time_res)
axsp.set_ylabel('Spike nr')
axsp.set_xlabel('Time[s]')
plf.spineless(axes, 'tr')
#plt.savefig('/media/owncloud/20181105_meeting_files/GLMsimulated.pdf',
#            bbox_inches='tight')
plt.show()



#%%
fig2, axes2 = plt.subplots(1, 1)
[axk] = np.array([axes2]).ravel()
axk.plot(t[:filter_length], k_real, label='Real filter')
#axk.plot(t[:filter_length], k_res/np.abs(k_res).max(), label='Predicted')
axk.plot(t[:filter_length], k_res, label='Predicted')
axk.set_xlabel('Time[s]')
axk.legend()
print(f'mu_real: {mu_real:4.2f}\nmu_res: {mu_res:4.2f}')
plf.spineless(axes2, 'tr')
#plt.savefig('/media/owncloud/20181105_meeting_files/GLMsimulated_filter.pdf',
#            bbox_inches='tight')
plt.show()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Implementing jacobian for GLM, to help troubleshoot GQM

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import plotfuncs as plf

import genlinmod as glm


filter_length = 40
frame_rate = 60
time_res = (1/frame_rate)
tstop = 500 # in seconds
t = np.arange(0, tstop, time_res)
np.random.seed(1221)

# Initialize model neuron
k_real = np.exp(-(t[:filter_length]-.14)**2/.002)-np.exp(-(t[:filter_length]-.17)**2/.001)
#k_real *= .5
mu_real = .01
f = glm.glm_fr(k_real, mu_real)

# Generate stimulus
x = np.random.normal(size=t.shape)

# Calculate the firing rate and spikes of the neuron given the stimulus
rate = f(x)
ratetimesdelta = True
if ratetimesdelta:
    rate *= time_res
spikes = np.random.poisson(rate)

np.random.seed()
k_guess = np.random.sample(size=filter_length)-.5
k_guess = np.zeros(filter_length)
mu_guess = spikes.mean()*time_res

#%%
debug_grad = False
usegrad = True
method = None

res = glm.minimize_loglhd(k_guess, mu_guess, x, time_res, spikes,
                          usegrad=usegrad,
                          debug_grad=debug_grad,
                          method=method,
                          options={'disp':True},
                          tol=1e-1,
                         )

#%%
if not debug_grad:
    k_res, mu_res = res['x'][:-1], res['x'][-1]
    fig2, axes2 = plt.subplots(1, 1)
    [axk] = np.array([axes2]).ravel()
    axk.plot(t[:filter_length], k_real, label='Real filter')
    #axk.plot(t[:filter_length], k_res/np.abs(k_res).max(), label='Predicted')
    axk.plot(t[:filter_length], k_res, label='Predicted')
    axk.set_xlabel('Time[s]')
    axk.legend()
    axk.text(.8, .6, f'mu_real: {mu_real:4.2f}\nmu_res: {mu_res:4.2f}',
             transform=axk.transAxes)
    axk.text(.98, .7, f'ratetimesdelta: {ratetimesdelta:}\nusegrad: {usegrad:}',
             transform=axk.transAxes, ha='right')
    plf.spineless(axes2, 'tr')
    #plt.savefig('/media/owncloud/20181105_meeting_files/GLMsimulated_filter.pdf',
    #            bbox_inches='tight')
    plt.show()
else:
    auto, manu = res
    kmu = [*k_real, mu_real]
    plt.plot(auto(kmu), label='auto')
    plt.plot(manu(kmu), label='manu')
    plt.legend()
    plt.title('Gradients')
    plt.show()

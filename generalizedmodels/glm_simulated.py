#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Implementing jacobian for GLM, to help troubleshoot GQM

"""

import numpy as np
import matplotlib.pyplot as plt
import plotfuncs as plf

import genlinmod as glm


filter_length = 40
frame_rate = 60
time_res = (1/frame_rate)
tstop = 500 # in seconds
t = np.arange(0, tstop, time_res)
np.random.seed(1221)

#%%
# Initialize model neuron
k_real = np.exp(-(t[:filter_length]-.14)**2/.002)-np.exp(-(t[:filter_length]-.17)**2/.001)

# Scale free k
filt_tmax = t[filter_length]
k_real_sf = (np.exp(-(t[:filter_length]-filt_tmax*.215)**2/(filt_tmax/200))
            -np.exp(-(t[:filter_length]-filt_tmax*.255)**2/(filt_tmax/400)))
#plt.plot(k_real); plt.plot(k_real_sf); plt.show()

k_real = k_real_sf
#%%
#k_real *= .5
mu_real = .7
f = glm.glm_fr(k_real, mu_real, time_res)

# Generate stimulus
x = np.random.normal(size=t.shape)

# Calculate the firing rate and spikes of the neuron given the stimulus
rate = f(x)

spikes = np.random.poisson(rate)

np.random.seed()
k_guess = np.random.sample(size=filter_length)-.5
k_guess = np.zeros(filter_length)
mu_guess = spikes.mean()*time_res

#%%
usegrad = True
method = None

res = glm.minimize_loglhd(k_guess, mu_guess, x, time_res, spikes,
                          usegrad=usegrad,
                          method=method,
                          options={'disp':True},
                          tol=1e-1,
                         )

#%%

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
axk.text(.98, .7, f'usegrad: {usegrad:}',
         transform=axk.transAxes, ha='right')
axk.set_ylim([-.8, 1.1])
plf.spineless(axes2, 'tr')
#plt.savefig('/media/owncloud/20181105_meeting_files/GLMsimulated_filter.pdf',
#            bbox_inches='tight')
plt.show()


plt.figure()
pred_fr = glm.glm_fr(k_res, mu_res, time_res)(x)

plt.plot(rate, lw=.6, label='Real firing rate')
plt.plot(pred_fr, lw=.6, label='Predicted firing rate')
#plt.plot(rate/time_res, lw=.6, label='rate/delta')
plt.show()

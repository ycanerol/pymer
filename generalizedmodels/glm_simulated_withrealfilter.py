#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""
import numpy as np
import matplotlib.pyplot as plt

import iofuncs as iof
import analysis_scripts as asc
import genlinmod as glm

expstim = ('20180802', 1)

data = iof.load(*expstim)
_, metadata = asc.read_spikesheet(expstim[0])
sta = data['stas'][3]


filter_length = sta.shape[0]
refresh_rate = metadata['refresh_rate']
time_res = 1/refresh_rate

t = np.arange(0, data['total_frames']*time_res, time_res)

k_in = sta
mu_in = 1.25

f = glm.glm_fr(k_in, mu_in, time_res)

userealstim = False

# Use either the actual stimulus in the experiment or a virtual one
if userealstim:
    x = glm.loadstim(*expstim)
else:
    np.random.seed(20190125)
    x = np.random.normal(size=data['total_frames'])
    np.random.seed()

stimfrac = .5
sl = slice(int(len(x)*stimfrac))
x = x[sl]
t = t[sl]

rate = f(x)
spikes = np.random.poisson(rate)

res = glm.minimize_loglhd(np.zeros(k_in.shape), 0, x,
                          time_res, spikes,
#                          usegrad=False,
                          method='Newton-CG',
                          options={'disp':True},
                          tol=1e-2)
#%%
k_out, mu_out = np.split(res.x, [filter_length])
ax1 = plt.gca()
ax1.plot(k_in, label='k_in')
ax1.plot(k_out, label='k_out')
ax1.text(0.96, 0.97, f'mu_in: {mu_in:4.2f}\nmu_out: {mu_out[0]:4.2f}',
         va='top', ha='right', transform=ax1.transAxes)
ax1.text(0.96, 0.85, f'userealstim: {userealstim:}',
         va='top', ha='right', transform=ax1.transAxes)
plt.show()
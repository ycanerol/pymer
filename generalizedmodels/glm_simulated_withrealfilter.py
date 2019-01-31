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
mu_in = 0.25

f = glm.glm_fr(k_in, mu_in, time_res)

userealstim = True

# Use either the actual stimulus in the experiment or a virtual one
if userealstim:
    x = glm.loadstim(*expstim)
else:
    np.random.seed(20190125)
    x = np.random.normal(size=data['total_frames'])
    np.random.seed()

stimfrac = 1
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
#ax1.text(0.96, 0.97, f'mu_in: {mu_in:4.2f}\nmu_out: {mu_out[0]:4.2f}',
#         va='top', ha='right', transform=ax1.transAxes)
ax1.plot(k_out.shape[0]+1, mu_in, 'o', label='mu_in', color='C0')
ax1.plot(k_out.shape[0]+1, mu_out, 'o', label='mu_out', color='C1')
ax1.legend(loc='lower right', ncol=2)
ax1.text(0.96, 0.85, f'userealstim: {userealstim:}',
         va='top', ha='right', transform=ax1.transAxes)
plt.title('Input and returned GLM parameters for simulation')
plt.show()

#%%
fr_out = glm.glm_fr(k_out, mu_out, time_res)(x)


plt.plot(rate, label='rate_in')
plt.plot(fr_out, label='rate_out', lw=.6)
plt.plot(rate-fr_out, label='in - out')
plt.legend(fontsize='x-small')
plt.title('Firing rate (Poisson process intensity)')
plt.show()

#%%
spikes_out = np.random.poisson(fr_out)

spikes_c = spikes.copy()
spikes_out_c = spikes_out.copy()

row_width = 200
nrows = (int(spikes.shape[0]/row_width))
spikes_c.resize(nrows, row_width)
spikes_out_c.resize(nrows, row_width)

im = plt.imshow(np.vstack((spikes_c, np.zeros(row_width)-1, spikes_out_c)),
                 vmin=-0.01, vmax=max(spikes.max(), spikes_out.max()))
plt.yticks
plt.grid(alpha=.4, axis='x')
im.cmap.set_under('r')
plt.axis('equal')
plt.show()

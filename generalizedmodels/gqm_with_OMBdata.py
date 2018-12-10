#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""

import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('/home/ycan/Documents/scripts/generalizedmodels/')

import gen_quad_model as gqm
import genlinmod as glm
import analysis_scripts as asc
import iofuncs as iof
from scipy import linalg


exp_name = '20180710'
stim_nr = 8
data = iof.load(exp_name, stim_nr)
stimulus_xy = glm.loadstim(exp_name, stim_nr)
stimulus = stimulus_xy[0, :]
clusters = data['clusters']

parameters = asc.read_parameters(exp_name, stim_nr)
_, frametimes = asc.ft_nblinks(exp_name, stim_nr, parameters.get('Nblinks', 2))

filter_length = l = data['filter_length']
refresh_rate = asc.read_spikesheet(exp_name)[1]['refresh_rate']

i = 0

# Limit the amount of data that will be loaded
lim = 15000

sta = data['stas'][i][0]
rawspikes = asc.read_raster(exp_name, stim_nr, *clusters[i][:2])[:lim]
frametimes = frametimes[:-1][:lim]
stimulus = stimulus[:lim]

spikes = asc.binspikes(rawspikes, frametimes)


import time
start = time.time()
res = gqm.minimize_loglikelihood(np.zeros(l), np.zeros((l, l)), 0,
                                 stimulus, 1/refresh_rate, spikes)
elapsed = time.time()-start

print(f'Time elapsed: {elapsed/60:6.1f} mins')
k_out, Q_out, mu_out = gqm.splitpars(res.x)


#%%
savepath = '/home/ycan/Documents/meeting_notes/2018-12-05/'
axk = plt.subplot(321)
axk.plot(sta, label='STA')
axk.plot(k_out, label='k_out')
axk.legend(fontsize='x-small')

axk.text(0.8, 0.5, f'mu_out: {mu_out:4.2f}', transform=axk.transAxes)

axq = plt.subplot(322)
axq.imshow(Q_out)

#plt.show()

w_out, v_out = linalg.eigh(Q_out)

colors = plt.rcParams['axes.prop_cycle'].by_key()['color'][2:]

axv = plt.subplot(323)
axw = plt.subplot(324)

axv.plot(w_out, 'ko')
eiginds = [0, 1, 19]
for ind, (eigind, w) in enumerate(zip(eiginds, w_out[eiginds])):
    axv.plot(eigind, w, 'o', color=colors[ind])
    axw.plot(v_out[:, eigind], lw=.8, color=colors[ind])
axw.plot(data['eigvecs_x'][i][::-1, -1], '--',label='STC0')
axw.legend(fontsize='x-small')

plt.tight_layout()
plt.savefig(savepath+'gqm_omb.pdf', bbox_inches = 'tight', pad_inches = 0.3)
plt.savefig(savepath+'gqm_omb.png', bbox_inches = 'tight', pad_inches = 0.3)
plt.show()

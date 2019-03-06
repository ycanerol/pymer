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
stim_nr = 1
data = iof.load(exp_name, stim_nr)
stimulus = glm.loadstim(exp_name, stim_nr)
clusters = data['clusters']
frametimes = asc.ft_nblinks(exp_name, stim_nr)[1]
filter_length = l = data['filter_length']
refresh_rate = asc.read_spikesheet(exp_name)[1]['refresh_rate']

i = 2
lim = None

sta = data['stas'][i]
rawspikes = asc.read_raster(exp_name, stim_nr, *clusters[i][:2])[:lim]
frametimes = frametimes[:lim]
stimulus = stimulus[:lim]
spikes = asc.binspikes(rawspikes, frametimes)

bin_length = np.ediff1d(frametimes).mean()
usegrad = True
method = 'Newton-CG'
minimize_disp = True

import time
start = time.time()
res = gqm.minimize_loglikelihood(np.zeros(l), np.zeros((l, l)), 0,
                                 stimulus, bin_length, spikes,
                                 usegrad=usegrad,
                                 method=method, minimize_disp=minimize_disp)
elapsed = time.time()-start

print(f'Time elapsed: {elapsed/60:6.1f} mins')
#%%

k_out, Q_out, mu_out = gqm.splitpars(res.x)

#savepath = '/home/ycan/Documents/meeting_notes/2018-12-05/'

fig = plt.figure(figsize=(7,5))
axk = plt.subplot(321)
axk.plot(sta, label='STA')
axk.plot(k_out, label='k_out')
axk.legend(fontsize='x-small')
axk.set_title('Linear filter')

axk.text(0.8, 0.5, f'mu_out: {mu_out:4.2f}', transform=axk.transAxes)

axq = plt.subplot(322)
axq.imshow(Q_out)
axq.set_title('Q out')


#plt.show()

w_out, v_out = linalg.eigh(Q_out)

colors = plt.rcParams['axes.prop_cycle'].by_key()['color'][2:]

axv = plt.subplot(323)
axw = plt.subplot(324)

axv.set_title('Eigenvalues of Q')

k_plotlim = filter_length

axv.plot(w_out, 'ko',)
eiginds = [0, 1, l-2, l-1]
for ind, (eigind, w) in enumerate(zip(eiginds, w_out[eiginds])):
    axv.plot(eigind, w, 'o', color=colors[ind])
    axw.plot(v_out[-k_plotlim:, eigind], lw=1, color=colors[ind])
axw.plot(sta[:l-1], '--', label='STA')
#axw.plot(k_out[k_plotlim::-1], '--')

axw.plot(data['eigvecs'][i][:l-1, -1], '--', label='STC0')
axw.legend(fontsize='x-small')

plt.tight_layout()
#plt.savefig(savepath+'gqm_fff.pdf', bbox_inches = 'tight', pad_inches = 0.3)
#plt.savefig(savepath+'gqm_fff.png', bbox_inches = 'tight', pad_inches = 0.3)
plt.show()

model = gqm.gqm_neuron(k_out, Q_out, mu_out,bin_length)(stimulus)
axfr = plt.subplot(311)
axfr.plot(model, color='C1')
axfr.set_title('GQM model, rate')
axsp = plt.subplot(312, sharex=axfr)
axsp.set_title('Spikes')
axsp.plot(spikes)
axspm = plt.subplot(313, sharex=axfr)
axspm.set_title('Model spikes')
axspm.plot(np.random.poisson(model))

plt.show()

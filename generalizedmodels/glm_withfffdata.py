#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt

import analysis_scripts as asc
import iofuncs as iof
import plotfuncs as plf

import genlinmod as glm
#from scratch_GLM2 import conv, minimize_loglhd
#from scratch_ste import loadstim

#def conv(k, x):
#    return np.convolve(k, x, 'full')[:-k.shape[0]+1]

def normalizestas(stas):
    stas = np.array(stas)
    b = np.abs(stas).max(axis=1)
    stas_normalized = stas / b.repeat(stas.shape[1]).reshape(stas.shape)
    return stas_normalized

#%%
exp_name = '20180802'
stim_nr = 1
data = iof.load(exp_name, stim_nr)
stimulus = glm.loadstim(exp_name, stim_nr)
clusters = data['clusters']
#%%
#stas = np.array(data['stas'])
#stas_normalized = np.abs(stas).max(axis=1)
#stas_normalized = a / stas_normalized.repeat(stas.shape[1]).reshape(stas.shape)
frametimes = asc.ft_nblinks(exp_name, stim_nr)[1]

#stas = normalizestas(data['stas'])
stas = np.array(data['stas'])

predstas = np.zeros(stas.shape)
predmus = np.zeros(stas.shape[0])
start = dt.datetime.now()

allspikes = np.zeros((stas.shape[0], frametimes.shape[0]), dtype=np.int8)

for i, cluster in enumerate(clusters):

    #cluster = data['clusters'][i]
    sta = data['stas'][i]


    frame_dur = data['frame_duration']

    spikes = asc.read_raster(exp_name, stim_nr, *cluster)
    spikes = asc.binspikes(spikes, frametimes)
    allspikes[i, :] = spikes

    res = glm.minimize_loglhd(np.zeros(sta.shape), 0, stimulus,
                              frame_dur, spikes)
    #%%
    k_pred = res['x'][:-1]
    mu_pred = res['x'][-1]

    predstas[i, :] = k_pred
    predmus[i] = mu_pred

    #ax1 = plt.subplot(111)
    #ax1.plot(k_pred)
    #ax1.plot(sta)
    #print(f'Predicted baseline firing rate: {mu_pred:4.2f}')
    #print(f'Average spike per time bin: {spikes.mean()/frame_dur:4.2f}')

#predstas = normalizestas(predstas)

elapsed = dt.datetime.now()-start
print(f'\nTook {elapsed.total_seconds()/60:4.2f} minutes')
#%%
lim = 50
#plt.figure(figsize=(5, 10))
imshowarg = {'cmap':'Greys_r'}
ax_stas = plt.subplot(1, 3, 1)
ax_stas.set_ylabel('Cells')
ax_stas.set_xlabel('Time [ms]')
ax_stas.set_xticklabels(['0', '0', '300', '450'])
ax_stas.imshow(stas[:lim],**imshowarg)
ax_stas.set_title('STAs')
ax_pred = plt.subplot(1, 3, 2)
ax_pred.imshow(predstas[:lim], **imshowarg)
ax_pred.set_title('Predicted with\n log-likelihood maximization', fontsize='x-small')
ax_diff = plt.subplot(1, 3, 3)
im = ax_diff.imshow((stas-predstas)[:lim], vmin=-1, vmax=1, **imshowarg)
ax_diff.set_title('Difference')
ax_diff.set_xticklabels([''])
ax_diff.set_yticklabels([''])
ax_pred.set_yticklabels([''])
ax_pred.set_xticklabels([''])
#plt.suptitle(f'{exp_name} {iof.getstimname(exp_name, stim_nr)}')
plt.subplots_adjust(top=.85)
#plt.savefig('/media/owncloud/20181105_meeting_files/likelihood_fff.pdf', bbox_inches='tight')

plt.show()
#%%
predspikes = np.zeros(allspikes.shape, dtype=np.int8)-1
for i in range (clusters.shape[0]):
    pred_fr = glm.glm_fr(predstas[i, :], predmus[i], frame_dur)(stimulus)
    predspikes[i] = np.random.poisson(pred_fr)
#    plt.plot(allspikes[i, :], lw=.6)
#    plt.plot(predspikes[i], lw=.6)
#    plt.title('Cell {:3.0f}\nmu: {:5.3f}'.format(i, predmus[i]))
#    plt.show()
#%%
# Use this with %matplotlib qt to be able to scroll around
sl = slice(5000, 6500)
#sl = slice(None)
imshowkwargs = {'vmin':0, 'vmax':allspikes.max(),
#                'cmap':'Greys_r',
                'cmap':'magma',
                }
plt.figure()
ax_rsp = plt.subplot(311)
ax_rsp.matshow(allspikes[:, sl], **imshowkwargs)

ax_psp = plt.subplot(312, sharex=ax_rsp, sharey=ax_rsp)
im = ax_psp.matshow(predspikes[:, sl], **imshowkwargs)
im.cmap.set_over('r')
im.cmap.set_under('k')
ax_stim = plt.subplot(313, sharex=ax_rsp)
ax_stim.plot(stimulus[sl], lw=.8)
plf.colorbar(im, ax=ax_stim, size='1%')
plt.tight_layout()
plt.show()


#%%
avgspikes = allspikes.mean(axis=1)
avgspikes_pred = predspikes.mean(axis=1)


plt.figure(figsize=(8.5, 5.5))
ax1 = plt.subplot(121)
#ax1 = plt.gca()
ax1.scatter(avgspikes, avgspikes_pred)
ax1.set_xlabel('Avg spike nr per time bin')
ax1.set_ylabel('Predicted spike nr per time bin')
ax1.set_xlim([-.05, .9])
ax1.set_ylim([-.05, .9])
ax1.plot([0, .9], [0, .9], 'r--', alpha=.7)
#plt.show()

#plt.figure()
#ax2 = plt.gca()
#ax2.scatter(avgspikes, predmus)
#ax2.set_xlabel('Avg spike nr per time bin')
#ax2.set_ylabel('Predicted mu')
#plt.show()


ax3 = plt.subplot(122)
#ax3 = plt.gca()
ax3.scatter(avgspikes, np.exp(predmus)*frame_dur)
ax3.set_xlabel('Avg spike nr per time bin')
ax3.set_ylabel('exp(mu)')
ax3.set_xlim([-.05, .9])
ax3.set_ylim([-.05, .9])
ax3.plot([0, .9], [0, .9], 'r--', alpha=.7)
plt.show()

#plt.figure()
#ax4 = plt.gca()
#ax4.scatter(avgspikes, np.log(predmus))
#ax4.set_xlabel('Avg spike nr per time bin')
#ax4.set_ylabel('log(mu)')
#plt.show()



#%%
import seaborn as sns

sns.jointplot(x=avgspikes, y=avgspikes_pred, kind='scatter',
              xlim=[0, .8], ylim=[0, .8],
              marginal_kws={'bins':20, 'rug':False})
plt.show()



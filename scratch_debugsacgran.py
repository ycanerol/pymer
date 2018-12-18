#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 11:43:07 2018

@author: ycan
"""
import numpy as np
import matplotlib.pyplot as plt
import iofuncs as iof
from saccadegratingsanalyzer import saccadegratingsanalyzer

#data = iof.load('Kara', 10)
#
#for key, val in data.items():
#    locals().update({key:val})
#
#
#for i in range(sacspikes.shape[0]):
#    plt.imshow(sacspikes[i, ...])
#    plt.show()
#
#spikemax = sacspikes.max()
##%%
#fig, axes = plt.subplots(4, 4, figsize=(8, 8))
#for j in range(4):
#    for k in range(4):
#        spikes = sacspikes[i, nton_sac[j][k], :]
##        ax = plt.gca()
#        ax = axes[3-j][k]
#        ax.imshow(spikes, vmin=0, vmax=spikemax)
#plt.show()

saccadegratingsanalyzer('Kara', 10)

#%%
mydata = dict(np.load('/home/ycan/Downloads/saccadegr_debugging.npz'))
hisdata = iof.readmat('/media/gruppenlaufwerk/FromPeopleToPeople/ToYunus/saccade_data.mat')
#%%
dfs = hisdata['fs']
dts = hisdata['ts']

mfs = mydata['stimpos']
mts = mydata['trans']
#%%
sh = np.squeeze(min(dfs.shape, mfs.shape))
fseq = np.all((dfs[:sh] == mfs [:sh]))
tseq = np.all((dts[:sh] == mts [:sh]))

#%%print('stim | trans')
print('m  d | m  d')
print('___________')
for i in range(20):
    print(f'{mfs[i]:<2} {int(dfs[i]):<1d} | {mts[i]:<2} {int(dts[i]):<1d}')
print(' =?  | =?  ')
print(f'{fseq} | {tseq}')

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 02:41:43 2018

@author: ycan
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import iofuncs as iof
import plotfuncs as plf
import analysis_scripts as asc
import texplot

cells = []
maxts = np.empty([2, 93])
for i, exp_name in enumerate(['20180118', '20180124', '20180207']):
    clusterids = plf.clusters_to_ids(asc.read_ods(exp_name)[0])
    cells.extend([(exp_name, cl_id) for cl_id in clusterids])
    if '20180124' in exp_name or '20180207' in exp_name:
        stripeflicker = [6, 12]
    elif '20180118' in exp_name:
        stripeflicker = [7, 14]
    a = np.empty([2, len(clusterids)])
    for j, stimnr in enumerate(stripeflicker):
        data=iof.load(exp_name, stimnr)
        a[j, :] = np.array(data['max_inds'])[:, 1]
#        maxts[j, :] = np.append(maxts[j, :], data['max_inds'])
    maxts = np.hstack((maxts, a))
maxts = maxts[:, 93:]
maxts = maxts*.0167*1000    # Convert to milliseconds
#%%
plt.plot(maxts)
plt.show()
#%%
plt.hist(maxts[1, :]-maxts[0, :])
plt.show()

#%%
data = np.load('/home/ycan/Documents/thesis/analysis_auxillary_files/'
               'thesis_csiplotting.npz')
cells = data['cells']
csi = data['csi']
include = data['include']
colors = data['colors']
colorcategories = data['colorcategories']

maxts_f = maxts[:, include]
#%%
fig = texplot.texfig(.9, 1.2)
#fig = plt.figure()
axes = fig.subplots(len(colorcategories)+1, 1, sharex=True)
bins = np.linspace(-.180, .025, 15)*1000
#bins = np.linspace(-.080, .025, 15)*1000
for i, color in enumerate(colorcategories):
    group = [index for index, c in enumerate(colors) if c == color]
    ax = axes[i]
    change = maxts_f[1, group]-maxts_f[0, group]
    ax.hist(change, color=color, bins=bins)
    ax.set_ylim([0, 20])
#    ax.set_yticks(np.linspace(0, 8, 3))
#    ax.set_alpha(0)
#    ax.plot([0, 0], [0, 12], 'r--', alpha=.3)
    plf.spineless(ax)
ax = axes[-1]
ax.hist(maxts_f[1, :]-maxts_f[0, :], color= 'k', bins=bins)
plf.spineless(ax)
ax.set_xlabel(r'Shift of temporal peak [ms]')
#plt.subplots_adjust(hspace=.3)
texplot.savefig('temporalpeakchangehistogram')
plt.show()

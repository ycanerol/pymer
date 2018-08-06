#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 02:41:43 2018

@author: ycan
"""
import numpy as np
import matplotlib.pyplot as plt
import iofuncs as iof
import plotfuncs as plf
import analysis_scripts as asc
from scipy import stats
import texplot


data = np.load('/home/ycan/Documents/thesis/analysis_auxillary_files/'
               'thesis_csiplotting.npz')
cells = []
include = data['include']
colors = data['colors']
colorcategories = data['colorcategories']
maxts = np.empty([2, 1])
for i, exp_name in enumerate(['20180118', '20180124', '20180207']):
    clusterids = plf.clusters_to_ids(asc.read_spikesheet(exp_name)[0])
    cells.extend([(exp_name, cl_id) for cl_id in clusterids])
    if '20180124' in exp_name or '20180207' in exp_name:
        stripeflicker = [6, 12]
    elif '20180118' in exp_name:
        stripeflicker = [7, 14]
    a = np.empty([2, len(clusterids)])
    for j, stimnr in enumerate(stripeflicker):
        data=iof.load(exp_name, stimnr)
        a[j, :] = np.array(data['max_inds'])[:, 1]
    maxts = np.hstack((maxts, a))
maxts = maxts[:, 1:] # Remove first element used in initialization
maxts = maxts*.0167*1000    # Convert to milliseconds

maxts_f = maxts[:, include]

fig = texplot.texfig(.9, .8)

axes = fig.subplots(len(colorcategories)+1, 1, sharex=True)
bins = np.arange(-200, 25+25, 25) # Arange does not include endpoint so 25+25
changes = []
for i, color in enumerate(colorcategories):
    group = [index for index, c in enumerate(colors) if c == color]
    ax = axes[i]
    change = maxts_f[1, group]-maxts_f[0, group]
    changes.append(change)
    ax.hist(change, color=color, bins=bins)
    ax.set_ylim([0, 10])
    plf.subplottext(['A', 'B', 'C', 'D', 'E'][i], ax, x=-.08)
    plf.spineless(ax)
    stat_test = stats.wilcoxon(change)
    print(f'Wilcoxon p-val for {color:25s}:  {stat_test[1]:7.2e}')
ax = axes[-1]
ax.hist(maxts_f[1, :]-maxts_f[0, :], color= 'k', bins=bins)
plf.subplottext('F', ax, x=-.08)
plf.spineless(ax)
ax.set_xlabel(r'TTP\textsubscript{photopic} - TTP\textsubscript{mesopic} [ms]')
plt.subplots_adjust(hspace=.4)
texplot.savefig('temporalpeakchangehistogram')
plt.show()

mannwu = stats.mannwhitneyu(changes[0], changes[1])
print(f'Mann-Whitney-U test p-val for ON vs OFF categories:  {mannwu[1]:7.2e}')

stats = stats.wilcoxon(maxts_f[1, :]-maxts_f[0, :])
print(f'Wilcoxon p-val for the whole population: {stats[1]:7.2e}')

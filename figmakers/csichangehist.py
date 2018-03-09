#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 15:38:19 2018

@author: ycan
"""
import numpy as np
import matplotlib.pyplot as plt
import texplot
import plotfuncs as plf

data = np.load('/home/ycan/Documents/thesis/analysis_auxillary_files/'
               'thesis_csiplotting.npz')
cells = data['cells']
csi = data['csi']
include = data['include']
colors = data['colors']
colorcategories = data['colorcategories']

fig = texplot.texfig(.9, 1.2)
axes = fig.subplots(len(colorcategories), 1, sharex=True)
csichange = csi[1, :] - csi[0, :]

bins = np.arange(-.2, .35+0.05, 0.05)
for i, color in enumerate(colorcategories):
    group = [index for index, c in enumerate(colors) if c == color]
    ax = axes[i]
    change = csichange[group]
    ax.hist(change, color=color , bins=bins)
    ax.set_ylim([0, 10])
    ax.set_yticks(np.linspace(0, 8, 3))
    ax.plot([0, 0], [0, 12], 'r--', alpha=.3)
    plf.subplottext(['A', 'B', 'C', 'D', 'E'][i], ax, x=-.08)
    plf.spineless(ax)

plt.subplots_adjust(hspace=.3)
texplot.savefig('csihistogram_mp')
plt.show()
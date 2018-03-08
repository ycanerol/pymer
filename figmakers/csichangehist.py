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
for i, color in enumerate(colorcategories):
    group = [index for index, c in enumerate(colors) if c == color]
    ax = axes[i]
    change = csi[1, group] - csi[0, group]
    ax.hist(change, color=color, bins=np.linspace(-.8, .8, 30))
    ax.set_ylim([0, 10])
    ax.set_xlim([-.2, .4])
    ax.set_yticks(np.linspace(0, 8, 3))
    ax.plot([0, 0], [0, 12], 'r--', alpha=.3)
    plf.spineless(ax)
plt.subplots_adjust(hspace=.3)
texplot.savefig('csihistogram_mp')
plt.show()

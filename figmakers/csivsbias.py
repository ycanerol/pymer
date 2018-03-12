#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 10:50:11 2018

@author: ycan
"""

import numpy as np
import matplotlib.pyplot as plt
import plotfuncs as plf
import texplot

data = np.load('/home/ycan/Documents/thesis/analysis_auxillary_files/'
               'thesis_csiplotting.npz')
cells = data['cells']
csi = data['csi']
bias = data['bias']
groups = data['groups']
colorcategories = data['colorcategories']

biaschange = bias[1, ] - bias[0, ]
csichange = csi[1, ] - csi[0, ]

scatterkwargs = {'linewidths':.7,
#                 'alpha':.8,
                 'edgecolor':'k'}
linekwargs = {'color':'k', 'alpha':.5, 'linestyle':'dashed', 'linewidth':1}

fig = texplot.texfig(.85, aspect=1.85)

ax = plt.subplot2grid((4, 3), (0, 0), colspan=3, rowspan=2)

for group, color in zip(groups, colorcategories):

    ax.scatter(csichange[group], biaschange[group], c=color, **scatterkwargs)
    ax.axhline(0, **linekwargs)
    ax.axvline(0, **linekwargs)
    plf.subplottext('A', ax, x=-0.05)
ax.set_xlabel(r'CSI$_{photopic}$ - CSI$_{mesopic}$')
ax.set_ylabel(r'PI$_{photopic}$ - PI$_{mesopic}$')


for i, (group, color) in enumerate(zip(groups, colorcategories)):
    ax = plt.subplot2grid((4,3), (2+int((np.round((i-1)/3))), i%3))
    ax.scatter(csichange, biaschange, c='grey')
    ax.scatter(csichange[group], biaschange[group], c=color, **scatterkwargs)
    ax.axhline(0, **linekwargs)
    ax.axvline(0, **linekwargs)
    plf.spineless(ax, 'tr')
    plf.subplottext(['B', 'C', 'D', 'E', 'F'][i], ax, x=-.25)

#axes[-1].set_axis_off()
plt.subplots_adjust(hspace=.45, wspace=.45)

texplot.savefig('csichangevsbiaschange')
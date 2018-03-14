#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 00:15:28 2018

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

#plt.figure(figsize=(6,6))
texplot.texfig(.8, 1)
ax3 = plt.subplot(111)
for color, group in zip(colorcategories, groups):
    ax3.plot(bias[:, group], color=color, linewidth=.4)
#plt.axis('equal')
ax3.set_xticks([0, 1])
ax3.set_xticklabels(['Mesopic', 'Photopic'])
ax3.set_ylabel('Polarity Index')
plf.spineless(ax3)

texplot.savefig('polarityindexchange')

plt.show()

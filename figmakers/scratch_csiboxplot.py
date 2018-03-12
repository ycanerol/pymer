#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 11 23:02:28 2018

@author: ycan
"""
import numpy as np
import matplotlib.pyplot as plt
import texplot
import plotfuncs as plf
from scipy import stats

data = np.load('/home/ycan/Documents/thesis/analysis_auxillary_files/'
               'thesis_csiplotting.npz')
cells = data['cells']
csi = data['csi']
include = data['include']
colors = data['colors']
colorcategories = data['colorcategories']


colorlabels = data['colorlabels']
ax = plt.subplot(111)
bplot = ax.boxplot(changes, labels=colorlabels, patch_artist=True);
plt.axhline(0, color='k', linestyle='dashed', alpha=.1)
ax.set_ylabel(r'CSI$_{photopic}$ - CSI$_{mesopic}$')
for patch, color in zip(bplot['boxes'], colorcategories):
    patch.set_facecolor(color)
for tick in ax.get_xticklabels():
    tick.set_rotation(45)
#texplot.savefig('csichange_boxplot')

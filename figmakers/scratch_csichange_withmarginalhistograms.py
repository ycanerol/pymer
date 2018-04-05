#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 19:58:10 2018

@author: ycan

Marginal histograms for scatter plot of Center-Surround Index
"""
import numpy as np
import matplotlib.pyplot as plt

import plotfuncs as plf

data = np.load('/home/ycan/Documents/thesis/analysis_auxillary_files/'
               'thesis_csiplotting.npz')
cells = data['cells']
csi = data['csi']
include = data['include']
colors = data['colors']
colorcategories = data['colorcategories']

from matplotlib.gridspec import GridSpec
fig = plt.figure(figsize=(8,8))
gs = GridSpec(10, 10)
ax_main = plt.subplot(gs[1:9, :9])
ax_x = plt.subplot(gs[0, :9],sharex=ax_main)
ax_y = plt.subplot(gs[1:9, 9],sharey=ax_main)
scatterkwargs = {'c':colors, 'alpha':.8, 'linewidths':.5,
                 'edgecolor':'k',
                 's':35}
ax_main.scatter(csi[0, :], csi[1, :], **scatterkwargs)

# Mark the example cells with an asterisk
asterixes = [(0.03443051,  0.19385925), # Example ON cell 20180207 03001
         (0.03238909,  0.29553824)] # Example OFF cell 20180118 23102
for asterix in asterixes:
    ax_main.text(*asterix, '*', color='k')

bins = np.linspace(0, 0.4, 9)
histkwargs = {'bins':bins, 'color':'k', 'alpha':.6}
ax_x.hist(csi[0, :], **histkwargs)
ax_y.hist(csi[1, :], orientation='horizontal', **histkwargs)
ax_x.set_axis_off()
ax_y.set_axis_off()
plf.spineless(ax_x)
plf.spineless(ax_y)
plt.show()
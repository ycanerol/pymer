#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 00:15:28 2018

@author: ycan
"""
import numpy as np
import matplotlib.pyplot as plt
import plotfuncs as plf

data = np.load('/home/ycan/Documents/thesis/analysis_auxillary_files/'
               'thesis_csiplotting.npz')
cells = data['cells']
csi = data['csi']
bias = data['bias']
groups = data['groups']
colorcategories = data['colorcategories']

fig, axes = plt.subplots(2, 1, sharex=True)
ax1, ax2 = axes.ravel()

#ax1 = plt.subplot(211)
#ax2 = plt.subplot(212)

bins = np.arange(-1, 1+.125, .125)

#ax1.set_xticks([])
ax2.set_xlabel('Polarity Index')
ax1.set_title('Mesopic')
ax2.set_title('Photopic')
plf.spineless(ax1, 'tr')
plf.spineless(ax2, 'tr')

ax1.hist(bias[0, :], bins=bins)
ax2.hist(bias[1, :], bins=bins)

plt.show()

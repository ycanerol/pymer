#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 18:23:52 2018

@author: ycan
"""
import numpy as np
import matplotlib.pyplot as plt
import iofuncs as iof
import plotfuncs as plf

def stripestim(exp_name):
    if '20180124' in exp_name or '20180207' in exp_name:
        stripeflicker = [6, 12]
    elif '20180118' in exp_name:
        stripeflicker = [7, 14]
    return stripeflicker

exps = ['20180118', '20180124', '20180207']

data = np.load('/home/ycan/Documents/thesis/analysis_auxillary_files/'
               'thesis_csiplotting.npz')
include = data['include']
cells = data['cells']
groups = data['groups']

all_fits = np.empty((*cells.shape, 73))

for exp in exps:
    stim = stripestim(exp)
    fits_m = np.array(iof.load(exp, stim[0])['fits'])
    fits_p = np.array(iof.load(exp, stim[1])['fits'])

p = plf.numsubplots(nrcells)
axes = plt.subplots(*p)[1].ravel()
for i in range(nrcells):
    ax = axes[i]
    ax.plot(fits_m[i, :]);
    ax.plot(fits_p[i, :]);
    plf.spineless(ax)
    ax.set_axis_off()
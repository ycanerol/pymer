#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 01:00:53 2018

@author: ycan

Compare on off bias change in different light conditions.
"""
import iofuncs as iof
import os
import matplotlib.pyplot as plt
import plotfuncs as plf
import numpy as np

exp_name = '20180124'
exp_dir = iof.exp_dir_fixer(exp_name)

onoffinds = np.zeros((3, 30))
for i, stim in enumerate([3, 8, 14]):
    onoffinds[i, :] = iof.load(exp_name, stim)['onoffbias']

#%%
labels = ['1_low', '2_high', '3_low']
plt.figure(figsize=(12, 10))
ax = plt.subplot(111)
plt.plot(labels, onoffinds)
plt.ylabel('On-Off Bias')
plt.title('On-Off Bias Change')
plf.spineless(ax)

plotsave = os.path.join(exp_dir, 'data_analysis', 'onoffbias')

#plt.savefig(plotsave+'.svg', format = 'svg', bbox_inches='tight')
#plt.savefig(plotsave+'.pdf', format = 'pdf', bbox_inches='tight')
plt.show()

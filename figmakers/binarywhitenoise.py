#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 11 23:45:45 2018

@author: ycan
"""
import numpy as np
import matplotlib.pyplot as plt
import plotfuncs as plf

s = 10
frames = 5
savedir = '/home/ycan/Downloads/whitenoisestim/'

np.random.seed(0)
for i in range(frames):
    plt.matshow(np.random.randint(0, 2, s*s).reshape(s,s), cmap='Greys');
    plt.xticks([])
    plt.yticks([])
    plf.spineless(plt.gca())
    plt.savefig(savedir+f'checker{i}.svg', bbox_inches='tight', pad_inches=0)

np.random.seed(0)

for i in range(frames):
    a = np.repeat(np.random.randint(0, 2, s), s)
    a = a.reshape(s,s)
    plt.imshow(a, cmap='Greys')
    plt.xticks([])
    plt.yticks([])
    plf.spineless(plt.gca())
    plt.savefig(savedir+f'stripe{i}.svg', bbox_inches='tight', pad_inches=0)
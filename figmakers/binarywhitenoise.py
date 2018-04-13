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

def setspines(ax):
    locs = ['top', 'left', 'bottom', 'right']
    for loc in locs:
        ax.spines[loc].set_color('orange')
        ax.spines[loc].set_linewidth(4)




np.random.seed(0)
fig = plt.figure()
for i in range(frames):
    d = i/15
    ax = fig.add_axes([.5-d, .5-d, .7, .7])
    ax.matshow(np.random.randint(0, 2, s*s).reshape(s,s), cmap='Greys');
    plt.xticks([])
    plt.yticks([])
    setspines(ax)
#    plf.spineless(plt.gca())
#    plt.savefig(savedir+f'checker{i}.svg', bbox_inches='tight', pad_inches=0)
plt.show()

np.random.seed(0)
fig = plt.figure()

for i in range(frames):
    d = i/15
    a = np.repeat(np.random.randint(0, 2, s), s)
    a = a.reshape(s,s)
    ax = fig.add_axes([.1+d, .1+d, .7, .7])
    ax.imshow(a, cmap='Greys')
    plt.xticks([])
    plt.yticks([])
    plf.spineless(plt.gca())
#    plt.savefig(savedir+f'stripe{i}.svg', bbox_inches='tight', pad_inches=0)
plt.show()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 17:35:44 2018

@author: ycan
"""
from randpy import randpy
import numpy as np
import matplotlib.pyplot as plt
import plotfuncs as plf

multiplier = 100

sy = 160
total_frames = 80000*multiplier
filter_length = 40
seed = -1000
length = sy*total_frames

randnrs, seed = randpy.ran1(seed, length)
randnrs = [1 if i > .5 else -1 for i in randnrs]

stimulus = np.reshape(randnrs, (sy, total_frames), order='F')
del randnrs

sta = np.zeros((sy, filter_length))
spikecounter = 0

for k in range(filter_length, total_frames-filter_length+1):
    stim_small = stimulus[:, k-filter_length+1:k+1][:, ::-1]
    spike = np.random.poisson()
    if spike != 0:
        sta += spike*stim_small
        spikecounter+=spike
del stimulus
sta = sta/spikecounter
ax = plt.subplot(111)
plf.stashow(sta, ax)
plt.show()
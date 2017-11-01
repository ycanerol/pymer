#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 31 17:37:58 2017

@author: ycan

Generates a stimulus array from the random seed. To be used if the stimulus
file already on disk is not long enough for some reason.
"""
#import sys
#sys.path.append('/Users/ycan/Documents/official/gottingen/\
#lab rotations/LR3 Gollisch/RandPy')
#sys.path.append('/Users/ycan/Documents/official/gottingen/lab rotations\
#/LR3 Gollisch/scripts/')

import numpy as np
import randpy
sx = 300
sy = 400

total_length = 6000

rnd_numbers, seed = randpy.ran1(seed, total_length*sx*sy)

# %% Reshape for checkerflicker
rnd_numbers = np.array(rnd_numbers).reshape(sx, sy, total_length, order='F')

rnd_numbers = np.array(np.where(rnd_numbers > .5, 1, -1), dtype='int8')

np.savez('/home/ycan/Documents/data/checker/c{}.npz'.format(i),
         rnd_numbers=rnd_numbers,
         seed = seed)

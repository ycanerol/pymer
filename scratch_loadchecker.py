#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 12:34:48 2017

@author: ycan
"""
import numpy as np

def loadchecker(sx = 300, sy=400, chunks=11, clength=6000):
    path = '/home/ycan/Documents/data/checker/'
    chunks = 11
    stimulus = np.empty((sx, sy, chunks*clength), dtype='int8')
    for i in range(chunks):
        a = np.load(path+'c{}.npz'.format(i))['rnd_numbers']
        stimulus[:, :, i*clength:(i+1)*clength] = a
        del a
    return stimulus

stimulus = loadchecker()

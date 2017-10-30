#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 09:33:23 2017

@author: ycan
"""
import numpy as np
import randpy

def loadstimulus(stimuluspath, sx, sy, totalframes):
    # Loads the pregenerated random numbers in appropriate length, reshapes
    # into current stimulus shape and returns the stimulus array.

    numbers = np.load(stimuluspath, mmap_mode='r')[:sx*sy*totalframes]
    stimulus = numbers.reshape(sx, sy, totalframes, order='F')

    #For checkerflicker,

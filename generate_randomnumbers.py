#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 11:36:29 2017

@author: ycan

Generate the random numbers that will be used in stimulus generation and save.

For experiments with a lot of random numbers, this step takes a long time. Read
ing from disk is faster than regenerating the numbers each time.

Compated to generate_stimulus.py, this one only outputs the numbers which then
need to be separately converted into the stimulus.

Of note, when generating the spatial structure stimuli the array needs to be in
fortran order like so:

np.array(rnd_numbers).reshape(sx, sy, total_frames, order='F')

"""
import numpy as np
import randpy


length = int(1.5e9)

rnd_numbers, seed = randpy.ran1(-10000, length)
rnd_numbers = np.array(rnd_numbers)




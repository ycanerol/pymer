#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 19:01:30 2018

@author: ycan
"""

import numpy as np
import scipy
import iofuncs as iof
import os

path = iof.exp_dir_fixer('Kara')
name = iof.stimname(path, 3)

"""
Extract frame times from .mat files. Needed for analyzing data
from other people, binary files containing the frame time pulses
are not usually available.


"""
matfile = os.path.join(path, 'frametimes', name + '_frametimings.mat')

try:
    f = scipy.io.matlab.loadmat(matfile)
    ftimes = f['ftimes'][0, :]
except NotImplementedError:
    import h5py
    with h5py.File(matfile, mode='r') as f:
        ftimes = f['ftimes'][:]
        if len(ftimes.shape) != 1:
            ftimes = ftimes.flatten()

ftimes = ftimes/1000

np.savez(os.path.join(path, 'frametimes', name + '_frametimes'),
         f_on=ftimes)

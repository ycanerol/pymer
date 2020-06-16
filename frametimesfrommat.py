#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 19:01:30 2018

@author: ycan
"""
import os
import numpy as np
import scipy.io
import iofuncs as iof
import analysis_scripts as asc


def frametimesfrommat(exp_name):
    """
    Extract frame times from .mat files. Needed for analyzing data
    from other people, binary files containing the frame time pulses
    are not usually available.

    The converted frametime files are corrected for monitor delay.
    """
    exp_dir = iof.exp_dir_fixer(exp_name)

    _, metadata = asc.read_spikesheet(exp_dir)
    monitor_delay = metadata['monitor_delay(s)']

    for i in range(1, 100):
        try:
            name = iof.getstimname(exp_dir, i)
        except IndexError as e:
            if str(e).startswith('Stimulus'):
                continue
            else:
                raise

        matfile = os.path.join(exp_dir, 'frametimes', name + '_frametimings.mat')
        # Check for zero padded name
        if not os.path.isfile(matfile):
            matfile = os.path.join(exp_dir, 'frametimes', '0' + name + '_frametimings.mat')

        try:
            f = scipy.io.matlab.loadmat(matfile)
            ftimes = f['ftimes'][0, :]
        except NotImplementedError:
            import h5py
            with h5py.File(matfile, mode='r') as f:
                ftimes = f['ftimes'][:]
                if len(ftimes.shape) != 1:
                    ftimes = ftimes.flatten()

        ftimes += monitor_delay

        np.savez(os.path.join(exp_dir, 'frametimes', name + '_frametimes'),
                 f_on=ftimes)
        print(f'Converted and saved frametimes for {name}')

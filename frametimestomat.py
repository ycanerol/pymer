#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 11:15:11 2018

@author: ycan
"""
import os.path
import scipy.io

from .modules import analysisfuncs as asc
from .modules import iofuncs as iof


def savenpztomat(exp_name, savedir=None):
    """
    Convert frametime files in .npz to .mat for interoperability
    with MATLAB users.

    savedir
    """
    exp_dir = iof.exp_dir_fixer(exp_name)

    _, metadata = asc.read_spikesheet(exp_dir)
    monitor_delay = metadata['monitor_delay(s)']

    for i in range(1, 100):
        print(i)
        try:
            ft_on, ft_off = asc.readframetimes(exp_name, i, returnoffsets=True)
        except ValueError as e:
            if str(e).startswith('No frametimes'):
                break
            else:
                raise
        # Convert to milliseconds b/c that is the convertion in MATLAB scripts
        ft_on = (ft_on - monitor_delay)*1000
        ft_off = (ft_off - monitor_delay)*1000

        stimname = iof.getstimname(exp_dir, i)

        if savedir is None:
            savedir = os.path.join(exp_dir, 'frametimes')
        savename = os.path.join(savedir, stimname)
        print(savename)
        scipy.io.savemat(savename+'_frametimings',
                         {'ftimes': ft_on,
                          'ftimes_offsets': ft_off},
                         appendmat=True)

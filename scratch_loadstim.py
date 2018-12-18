#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 16:36:22 2018

@author: ycan
"""
import numpy as np
import analysis_scripts as asc
from randpy import randpy

def loadstim(exp, stim_nr):
    sortedstim = asc.stimulisorter(exp)
    clusters, metadata = asc.read_spikesheet(exp)

    for key, val in sortedstim.items():
        if stim_nr in val:
            stimtype = key
    if stimtype in ['fff', 'stripeflicker', 'checkerflicker']:
        pars = asc.read_parameters(exp, stim_nr)
        seed = pars.get('seed', -10000)
        bw = pars.get('blackwhite', False)
        nblinks = pars.get('Nblinks', None)
        refresh_rate = metadata['refresh_rate']
        filter_length, frametimings = asc.ft_nblinks(exp, stim_nr)
        total_frames = frametimings.shape[0]
        total_frames = 1000

        if stimtype == 'fff':
            if bw:
                randnrs, seed = randpy.ranb(seed, total_frames)
                # Since ranb returns zeros and ones, we need to convert
                # the zeros into -1s.
                stimulus = np.array(randnrs) * 2 - 1
            else:
                randnrs, seed = randpy.gasdev(seed, total_frames)
                stimulus = np.array(randnrs)
        elif stimtype == 'checkerflicker':
            scr_width = metadata['screen_width']
            scr_height = metadata['screen_height']
            stx_h = pars['stixelheight']
            stx_w = pars['stixelwidth']
            # Check whether any parameters are given for margins, calculate
            # screen dimensions.
            marginkeys = ['tmargin', 'bmargin', 'rmargin', 'lmargin']
            margins = []
            for key in marginkeys:
                margins.append(pars.get(key, 0))
            # Subtract bottom and top from vertical dimension; left and right
            # from horizontal dimension
            scr_width = scr_width-sum(margins[2:])
            scr_height = scr_height-sum(margins[:2])
            sx, sy = scr_height/stx_h, scr_width/stx_w
            # Make sure that the number of stimulus pixels are integers
            # Rounding down is also possible but might require
            # other considerations.
            if sx % 1 == 0 and sy % 1 == 0:
                sx, sy = int(sx), int(sy)
            else:
                raise ValueError('sx and sy must be integers')

            randnrs, seed = randpy.ranb(seed, sx*sy*total_frames)
            # Reshape and change 0's to -1's
            stimulus = np.reshape(randnrs, (sx, sy, total_frames), order='F')*2-1
        return stimulus
    return None

#a = loadstim('20180710', 1)
#b = loadstim('20180207', 5)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 14:36:31 2018

@author: ycan
"""

import numpy as np
from os.path import join as pjoin
import os
import analysis_scripts as asc
import iofuncs as iof
from randpy import randpy


def stripeflickeranalysis(exp_name, stim_nrs):
    exp_dir = iof.exp_dir_fixer(exp_name)

    if isinstance(stim_nrs, int):
        stim_nrs = [stim_nrs]

    for stim_nr in stim_nrs:
        stimname = iof.getstimname(exp_name, stim_nr)

        clusters, metadata = asc.read_ods(exp_dir)

        parameters = asc.read_parameters(exp_dir, stim_nr)

        scr_width = metadata['screen_width']
        px_size = metadata['pixel_size(um)']

        stx_w = parameters['stixelwidth']
        stx_h = parameters['stixelheight']

        if (stx_h/stx_w) < 2:
            raise ValueError('Make sure the stimulus is stripeflicker.')

        sy = scr_width/stx_w
        if sy % 1 == 0:
            sy = int(sy)
        else:
            raise ValueError('sy is not an integer')

        nblinks = parameters['Nblinks']
        try:
            bw = parameters['blackwhite']
        except KeyError:
            bw = False

        try:
            seed = parameters['seed']
        except KeyError:
            seed = -10000

        if nblinks == 1:
            ft_on, ft_off = asc.readframetimes(exp_dir, stim_nr,
                                               returnoffsets=True)
            # Initialize empty array twice the size of one of them, assign
            # value from on or off to every other element.
            frametimings = np.empty(ft_on.shape[0]*2, dtype=float)
            frametimings[::2] = ft_on
            frametimings[1::2] = ft_off
            # Set filter length so that temporal filter is ~600 ms.
            # The unit here is number of frames.
            filter_length = 40
        elif nblinks == 2:
            frametimings = asc.readframetimes(exp_dir, stim_nr)
            filter_length = 20
        else:
            raise ValueError('Unexpected value for nblinks.')

        frame_duration = np.average(np.ediff1d(frametimings))
        total_frames = frametimings.shape[0]

        all_spiketimes = []
        # Store spike triggered averages in a list containing correct
        # shaped arrays
        stas = []

        for i in range(len(clusters[:, 0])):
            spiketimes = asc.read_raster(exp_dir, stim_nr,
                                         clusters[i, 0], clusters[i, 1])

            spikes = asc.binspikes(spiketimes, frametimings)
            all_spiketimes.append(spikes)
            stas.append(np.zeros((sy, filter_length)))
                # If max_i cannot be found just set it to zeros.

        if bw:
            randnrs, seed = randpy.ran1(seed, sy*total_frames)
            randnrs = [1 if i > .5 else -1 for i in randnrs]
        else:
            randnrs, seed = randpy.gasdev(seed, sy*total_frames)

        stimulus = np.reshape(randnrs, (sy, total_frames), order='F')
        del randnrs

        for k in range(filter_length, total_frames-filter_length+1):
            stim_small = stimulus[:, k-filter_length+1:k+1][:, ::-1]
            for j in range(clusters.shape[0]):
                spikes = all_spiketimes[j]
                if spikes[k] != 0:
                    stas[j] += spikes[k]*stim_small

        max_inds = []
        spikenrs = np.array([spikearr.sum() for spikearr in all_spiketimes])

        quals = np.array([])

        for i in range(clusters.shape[0]):
            stas[i] = stas[i]/spikenrs[i]
            # Find the pixel with largest absolute value
            max_i = np.squeeze(np.where(np.abs(stas[i])
                                        == np.max(np.abs(stas[i]))))
            # If there are multiple pixels with largest value,
            # take the first one.
            if max_i.shape != (2,):
                try:
                    max_i = max_i[:, 0]
                # If max_i cannot be found just set it to zeros.
                except IndexError:
                    max_i = np.array([0, 0])

            max_inds.append(max_i)

            quals = np.append(quals, asc.staquality(stas[i]))

        savefname = str(stim_nr)+'_data'
        savepath = pjoin(exp_dir, 'data_analysis', stimname)

        exp_name = os.path.split(exp_dir)[-1]

        if not os.path.isdir(savepath):
            os.makedirs(savepath, exist_ok=True)
        savepath = os.path.join(savepath, savefname)

        keystosave = ['stas', 'max_inds', 'clusters', 'sy',
                      'frame_duration', 'all_spiketimes', 'stimname',
                      'total_frames', 'stx_w', 'spikenrs', 'bw',
                      'quals', 'nblinks', 'filter_length', 'exp_name']
        data_in_dict = {}
        for key in keystosave:
            data_in_dict[key] = locals()[key]

        np.savez(savepath, **data_in_dict)
        print(f'Analysis of {stimname} completed.')

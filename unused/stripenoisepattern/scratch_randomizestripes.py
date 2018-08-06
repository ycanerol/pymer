#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 14:36:31 2018

@author: ycan

Using to check the horizontal stripes
"""

import numpy as np
from os.path import join as pjoin
import os
import analysis_scripts as asc
import iofuncs as iof
from randpy import randpy
import matplotlib.pyplot as plt
import plotfuncs as plf

import random

def mersennetw(nr, seed1=-1000):
    random.seed(seed1)
    a = []
    [a.append(random.random()) for i in range(nr)]
    return np.array(a)

def corrector(sy, total_frames, filter_length, seed):
    print(f'corrector parameters: {sy}, {total_frames}, '
          f'{filter_length}, {seed}')
    sta = np.zeros((sy, filter_length))
    spikect = 0

    # Initialize the stimulus
    randnrs, seed = randpy.ran1(seed, sy*filter_length)
    randnrs = [1 if i > .5 else -1 for i in randnrs]
    stim = np.reshape(randnrs, (sy, filter_length), order='F')

    for frame in range(total_frames):
        randnrs, seed = randpy.ran1(seed, sy)
        randnrs = [1 if i > .5 else -1 for i in randnrs]
        stim = np.hstack((stim[:, 1:], np.array(randnrs)[..., None]))
        spike = np.random.poisson()
        spike = 1
        if spike != 0:
            sta += stim*spike
            spikect += spike
    sta /= spikect
    bar = sta[:, -1]
    return bar


def randomizestripes(label, exp_name='20180124', stim_nrs=6):
    exp_dir = iof.exp_dir_fixer(exp_name)

    if isinstance(stim_nrs, int):
        stim_nrs = [stim_nrs]

    for stim_nr in stim_nrs:
        stimname = iof.getstimname(exp_name, stim_nr)

        clusters, metadata = asc.read_spikesheet(exp_dir)

        parameters = asc.read_parameters(exp_dir, stim_nr)

        scr_width = metadata['screen_width']
        px_size = metadata['pixel_size(um)']

        stx_w = parameters['stixelwidth']
        stx_h = parameters['stixelheight']

        if (stx_h/stx_w) < 2:
            raise ValueError('Make sure the stimulus is stripeflicker.')

        sy = scr_width/stx_w
#        sy = sy*4
        sy = int(sy)

        nblinks = parameters['Nblinks']
        try:
            bw = parameters['blackwhite']
        except KeyError:
            bw = False

        try:
            seed = parameters['seed']
            initialseed = parameters['seed']
        except KeyError:
            seed = -10000
            initialseed = -10000

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

        # Omit everything that happens before the first 10 seconds
        cut_time = 10

        frame_duration = np.average(np.ediff1d(frametimings))
        total_frames = int(frametimings.shape[0]/4)

        all_spiketimes = []
        # Store spike triggered averages in a list containing correct
        # shaped arrays
        stas = []

        for i in range(len(clusters[:, 0])):
            spikes_orig = asc.read_raster(exp_dir, stim_nr,
                                         clusters[i, 0], clusters[i, 1])
            spikesneeded = spikes_orig.shape[0]*1000

            spiketimes = np.random.random_sample(spikesneeded)*spikes_orig.max()
            spiketimes = np.sort(spiketimes)
            spikes = asc.binspikes(spiketimes, frametimings)
            all_spiketimes.append(spikes)
            stas.append(np.zeros((sy, filter_length)))

        if bw:
            randnrs, seed = randpy.ran1(seed, sy*total_frames)
#            randnrs = mersennetw(sy*total_frames, seed1=seed)
            randnrs = [1 if i > .5 else -1 for i in randnrs]
        else:
            randnrs, seed = randpy.gasdev(seed, sy*total_frames)

        stimulus = np.reshape(randnrs, (sy, total_frames), order='F')
        del randnrs

        for k in range(filter_length, total_frames-filter_length+1):
            stim_small = stimulus[:, k-filter_length+1:k+1][:, ::-1]
            for j in range(clusters.shape[0]):
                spikes = all_spiketimes[j]
                if spikes[k] != 0 and frametimings[k]>cut_time:
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

#        savefname = str(stim_nr)+'_data'
#        savepath = pjoin(exp_dir, 'data_analysis', stimname)
#
#        exp_name = os.path.split(exp_dir)[-1]
#
#        if not os.path.isdir(savepath):
#            os.makedirs(savepath, exist_ok=True)
#        savepath = os.path.join(savepath, savefname)
#
#        keystosave = ['stas', 'max_inds', 'clusters', 'sy',
#                      'frame_duration', 'all_spiketimes', 'stimname',
#                      'total_frames', 'stx_w', 'spikenrs', 'bw',
#                      'quals', 'nblinks', 'filter_length', 'exp_name']
#        data_in_dict = {}
#        for key in keystosave:
#            data_in_dict[key] = locals()[key]
#
#        np.savez(savepath, **data_in_dict)
#        print(f'Analysis of {stimname} completed.')


        clusterids = plf.clusters_to_ids(clusters)

#        assert(initialseed.ty)
        correction = corrector(sy, total_frames, filter_length, initialseed)
        correction = np.outer(correction, np.ones(filter_length))

        t = np.arange(filter_length)*frame_duration*1000
        vscale = int(stas[0].shape[0] * stx_w*px_size/1000)
        for i in range(clusters.shape[0]):
            sta = stas[i]-correction

            vmax = 0.03
            vmin = -vmax
            plt.figure(figsize=(6, 15))
            ax = plt.subplot(111)
            im = ax.imshow(sta, cmap='RdBu', vmin=vmin, vmax=vmax,
                           extent=[0, t[-1], -vscale, vscale], aspect='auto')
            plt.xlabel('Time [ms]')
            plt.ylabel('Distance [mm]')

            plf.spineless(ax)
            plf.colorbar(im, ticks=[vmin, 0, vmax], format='%.2f', size='2%')
            plt.suptitle('{}\n{}\n'
                         '{} Rating: {}\n'
                         'nrofspikes {:5.0f}'.format(exp_name,
                                                       stimname,
                                                       clusterids[i],
                                                       clusters[i][2],
                                                       spikenrs[i]))
            plt.subplots_adjust(top=.90)
            savepath = os.path.join(exp_dir, 'data_analysis',
                                    stimname, 'STAs_randomized')
            svgpath = pjoin(savepath, label)
            if not os.path.isdir(svgpath):
                os.makedirs(svgpath, exist_ok=True)
            plt.savefig(os.path.join(svgpath, clusterids[i]+'.svg'),
                        bbox_inches='tight')
            plt.close()

    os.system(f"convert -delay 25 {svgpath}/*svg {savepath}/animated_{label}.gif")

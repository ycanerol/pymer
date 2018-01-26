#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 13:48:31 2018

@author: ycan
"""
import iofuncs as iof
import matplotlib.pyplot as plt
import plotfuncs as plf
import numpy as np
import analysis_scripts as asc
import os


def plotstripestas(exp_name, stim_nrs):
    """
    Plot and save all the STAs from multiple stripe flicker stimuli.
    """
    exp_dir = iof.exp_dir_fixer(exp_name)

    _, metadata = asc.read_ods(exp_dir)
    px_size = metadata['pixel_size(um)']

    for stim_nr in stim_nrs:
        data = iof.load(exp_name, stim_nr)

        clusters = data['clusters']
        stas = data['stas']
        max_inds = data['max_inds']
        filter_length = data['filter_length']
        stx_w = data['stx_w']
        exp_name = data['exp_name']
        stimname = data['stimname']
        frame_duration = data['frame_duration']
        quals = data['quals']

        clusterids = plf.clusters_to_ids(clusters)

        # Determine frame size so that the total frame covers
        # an area large enough i.e. 2*700um
        fsize = int(700/(stx_w*px_size))
        t = np.arange(filter_length)*frame_duration*1000
        vscale = fsize * stx_w*px_size
        for i in range(clusters.shape[0]):
            sta = stas[i]
            max_i = max_inds[i]
            try:
                sta, max_i = cutstripe(sta, max_i, fsize)
            except ValueError:
                continue
            vmax = np.max(np.abs(sta))
            vmin = -vmax
            plt.figure(figsize=(6, 6))
            ax = plt.subplot(111)
            im = ax.imshow(sta, cmap='RdBu', vmin=vmin, vmax=vmax,
                           extent=[0, t[-1], -vscale, vscale], aspect='auto')
            plt.xlabel('Time [s]')
            plt.ylabel('Distance[µm]')

            plf.spineless(ax)
            plf.colorbar(im, ticks=[vmin, 0, vmax], format='%.2f', size='2%')
            plt.suptitle('{}\n{}\n'
                         '{} Rating: {}\n'
                         'STA quality: {:4.2f}'.format(exp_name,
                                                       stimname,
                                                       clusterids[i],
                                                       clusters[i][2],
                                                       quals[i]))
            plt.subplots_adjust(top=.85)
            savepath = os.path.join(exp_dir, 'data_analysis',
                                    stimname, 'STAs')
            if not os.path.isdir(savepath):
                os.makedirs(savepath, exist_ok=True)
            plt.savefig(os.path.join(savepath, clusterids[i]+'.svg'))
            plt.close()
        print(f'Plotting of {stimname} completed.')


def cutstripe(sta, max_i, fsize):
    if max_i[0] - fsize <= 0 or max_i[0] + fsize > sta.shape[0]:
        raise ValueError('Cutting outside the STA range.')
    sta_r = sta[max_i[0]-fsize:max_i[0]+fsize+1, :]
    max_i_r = np.append(fsize, max_i[-1])
    return sta_r, max_i_r

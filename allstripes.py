#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 15:46:45 2018

@author: ycan

Plot multiple stripeflicker STAs together to see the
effect of parameters.

"""
import os
import iofuncs as iof
import matplotlib.pyplot as plt
import plotfuncs as plf
import numpy as np
import analysis_scripts as asc


def cutstripe(sta, max_i, fsize):
    if max_i[0] - fsize <= 0 or max_i[0] + fsize > sta.shape[0]:
        raise ValueError('Cutting outside the STA range.')
    sta_r = sta[max_i[0]-fsize:max_i[0]+fsize+1, :]
    max_i_r = np.append(fsize, max_i[-1])
    return sta_r, max_i_r


exp_name = '20171122'
stripes = [8, 9, 10, 11]

exp_dir = iof.exp_dir_fixer(exp_name)

data = iof.load(exp_name, stripes[0])


_, metadata = asc.read_ods(exp_dir)
px_size = metadata['pixel_size(um)']
exp_name = data['exp_name']
stx_w = data['stx_w']


clusters = data['clusters']
clusterids = plf.clusters_to_ids(clusters)

fsize = int(700/(stx_w*px_size))
vscale = fsize * stx_w*px_size


for i in range(len(clusterids)):
    plt.figure(figsize=(8, 8))

    for j, stripe in enumerate(stripes):
        data = iof.load(exp_name, stripe)
        stas = data['stas']
        max_inds = data['max_inds']
        filter_length = data['filter_length']
        frame_duration = data['frame_duration']
        quals = data['quals']

        sta = stas[i]
        max_i = max_inds[i]

        t = np.arange(filter_length)*frame_duration*1000

        try:
            sta, max_i = cutstripe(sta, max_i, fsize)
        except ValueError:
            continue
        vmax = np.max(np.abs(sta))
        vmin = -vmax
        ax = plt.subplot(2, 2, j+1)
        plt.title('STA quality: {:4.2f}'.format(quals[i]))

        ax.set_aspect('equal')

        im = ax.imshow(sta, cmap='RdBu', vmin=vmin, vmax=vmax,
                       extent=[0, t[-1], -vscale, vscale], aspect='auto')
        if j >= 2:
            plt.xlabel('Time [s]\n\nFrame duration: {:2.1f}'
                       'ms'.format(frame_duration*1000))
        else:
            ax.axes.get_xaxis().set_visible(False)
        if j == 0:
            plt.ylabel('B&W\n\nDistance[µm]')
        if j == 2:
            plt.ylabel('Gaussian\n\nDistance[µm]')
        if j % 2 == 1:
            ax.axes.get_yaxis().set_visible(False)

        plf.spineless(ax)
        plf.colorbar(im, ticks=[vmin, 0, vmax], format='%.2f', size='2%')
    plt.subplots_adjust(hspace=.4, wspace=.5)
    plt.suptitle('{}\n{}'.format(exp_name, clusterids[i]))

    savepath = os.path.join(exp_dir, 'data_analysis', 'allstripes')
    if not os.path.isdir(savepath):
        os.makedirs(savepath, exist_ok=True)
    plt.savefig(os.path.join(savepath, clusterids[i]+'.svg'))
    plt.close()

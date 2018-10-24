#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 19:00:09 2017

@author: ycan
"""
import matplotlib.pyplot as plt
import numpy as np
import os

from .modules import analysisfuncs as asc
from .modules import iofuncs as iof
from .modules import plotfuncs as plf


def plotcheckerstas(exp_name, stim_nr, filename=None):
    """
    Plot and save all STAs from checkerflicker analysis. The plots
    will be saved in a new folder called STAs under the data analysis
    path of the stimulus.

    <exp_dir>/data_analysis/<stim_nr>_*/<stim_nr>_data.h5 file is
    used by default. If a different file is to be used, filename
    should be supplied.
    """

    from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

    exp_dir = iof.exp_dir_fixer(exp_name)
    stim_nr = str(stim_nr)
    if filename:
        filename = str(filename)

    _, metadata = asc.read_spikesheet(exp_dir)
    px_size = metadata['pixel_size(um)']

    if not filename:
        savefolder = 'STAs'
        label = ''
    else:
        label = filename.strip('.npz')
        savefolder = 'STAs_' + label

    data = iof.load(exp_name, stim_nr, fname=filename)

    clusters = data['clusters']
    stas = data['stas']
    filter_length = data['filter_length']
    stx_h = data['stx_h']
    exp_name = data['exp_name']
    stimname = data['stimname']

    for j in range(clusters.shape[0]):
        a = stas[j]
        subplot_arr = plf.numsubplots(filter_length)
        sta_max = np.max(np.abs([np.max(a), np.min(a)]))
        sta_min = -sta_max
        plt.figure(dpi=250)
        for i in range(filter_length):
            ax = plt.subplot(subplot_arr[0], subplot_arr[1], i+1)
            im = ax.imshow(a[:, :, i], vmin=sta_min, vmax=sta_max,
                           cmap=iof.config('colormap'))
            ax.set_aspect('equal')
            plt.axis('off')
            if i == 0:
                scalebar = AnchoredSizeBar(ax.transData,
                                           10, '{} µm'.format(10*stx_h
                                                              * px_size),
                                           'lower left',
                                           pad=0,
                                           color='k',
                                           frameon=False,
                                           size_vertical=1)
                ax.add_artist(scalebar)
            if i == filter_length-1:
                plf.colorbar(im, ticks=[sta_min, 0, sta_max], format='%.2f')
        plt.suptitle('{}\n{}\n'
                     '{:0>3}{:0>2} Rating: {}'.format(exp_name,
                                                      stimname+label,
                                                      clusters[j][0],
                                                      clusters[j][1],
                                                      clusters[j][2]))

        savepath = os.path.join(exp_dir, 'data_analysis', stimname,
                                savefolder,
                                '{:0>3}{:0>2}'.format(clusters[j][0],
                                                      clusters[j][1]))

        os.makedirs(os.path.split(savepath)[0], exist_ok=True)

        plt.savefig(savepath+'.png', bbox_inches='tight')
        plt.close()
    print(f'Plotted checkerflicker STA for {stimname}')

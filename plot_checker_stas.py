#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 19:00:09 2017

@author: ycan
"""
import numpy as np
import matplotlib.pyplot as plt
import plotfuncs as plf
import iofuncs as iof
import analysis_scripts as asc
import glob
import os


def plot_checker_stas(exp_dir, stim_nr, filename=None):
    """
    Plot and save all STAs from checkerflicker analysis. The plots
    will be saved in a new folder called STAs under the data analysis
    path of the stimulus.

    <exp_dir>/data_analysis/<stim_nr>_*/<stim_nr>_data.h5 file is
    used by default. If a different file is to be used, filename
    should be supplied.
    """

    from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

    stim_nr = str(stim_nr)
    if filename:
        filename = str(filename)
    parent_path = glob.glob(os.path.join(exp_dir, 'data_analysis',
                                         stim_nr+'_*'))[0]
    _, metadata = asc.read_ods(exp_dir)
    px_size = metadata['pixel_size(um)']

    if not filename:
        datafile = os.path.join(parent_path, stim_nr+'_data.h5')
        savefolder = 'STAs'
        label = ''
    else:
        datafile = os.path.join(parent_path, filename)
        label = filename.strip('.h5')
        savefolder = 'STAs_' + label

    data = iof.loadh5(datafile)

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
                           cmap='RdBu')
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
                plt.colorbar(im)
        plt.suptitle('{}\n{}\n'
                     '{:0>3}{:0>2} Rating: {}'.format(exp_name,
                                                      stimname+label,
                                                      clusters[j][0],
                                                      clusters[j][1],
                                                      clusters[j][2]))
        savepath = os.path.join(parent_path,
                                savefolder,
                                '{:0>3}{:0>2}'.format(clusters[j][0],
                                                      clusters[j][1]))
        os.makedirs(os.path.split(savepath)[0], exist_ok=True)

        plt.savefig(savepath+'.png')
        plt.close()

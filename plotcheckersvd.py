#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 15:54:05 2018

@author: ycan
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import analysis_scripts as asc
import iofuncs as iof
import miscfuncs as msc
import plotfuncs as plf
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar


def plotcheckersvd(expname, stimnr, filename=None):
    """
    Plot the first two components of SVD analysis.
    """
    if filename:
        filename = str(filename)

    exp_dir = iof.exp_dir_fixer(expname)
    _, metadata = asc.read_ods(exp_dir)
    px_size = metadata['pixel_size(um)']

    if not filename:
        savefolder = 'SVD'
        label = ''
    else:
        label = filename.strip('.npz')
        savefolder = 'SVD_' + label

    data = iof.load(expname, stimnr, filename)

    stas = data['stas']
    max_inds = data['max_inds']
    clusters = data['clusters']
    stx_h = data['stx_h']
    frame_duration = data['frame_duration']
    stimname = data['stimname']
    exp_name = data['exp_name']

    clusterids = plf.clusters_to_ids(clusters)

    # Determine frame size so that the total frame covers
    # an area large enough i.e. 2*700um
    f_size = int(700/(stx_h*px_size))

    for i in range(clusters.shape[0]):
        sta = stas[i]
        max_i = max_inds[i]

        try:
            sta, max_i = msc.cut_around_center(sta, max_i, f_size=f_size)
        except ValueError:
            continue
        fit_frame = sta[:, :, max_i[2]]

        try:
            sp1, sp2, t1, t2, _, _ = msc.svd(sta)
        # If the STA is noisy (msc.cut_around_center produces an empty array)
        # SVD cannot be calculated, in this case we skip that cluster.
        except np.linalg.LinAlgError:
            continue

        plotthese = [fit_frame, sp1, sp2]

        plt.figure(dpi=200)
        plt.suptitle(f'{exp_name}\n{stimname}\n{clusterids[i]}')
        rows = 2
        cols = 3

        vmax = np.max(np.abs([sp1, sp2]))
        vmin = -vmax

        for j in range(len(plotthese)):
            ax = plt.subplot(rows, cols, j+1)
            im = plt.imshow(plotthese[j], vmin=vmin, vmax=vmax,
                            cmap='RdBu')
            ax.set_aspect('equal')
            plt.xticks([])
            plt.yticks([])
            for child in ax.get_children():
                if isinstance(child, matplotlib.spines.Spine):
                    child.set_color('C{}'.format(j % 3))
                    child.set_linewidth(2)
            if j == 0:
                plt.title('center px')
            elif j == 1:
                plt.title('SVD spatial 1')
            elif j == 2:
                plt.title('SVD spatial 2')
                plf.colorbar(im,
                             ticks=[vmin, 0, vmax],
                             format='%.2f')
                barsize = 100/(stx_h*px_size)
                scalebar = AnchoredSizeBar(ax.transData,
                                           barsize, '100 µm',
                                           'lower left',
                                           pad=0,
                                           color='k',
                                           frameon=False,
                                           size_vertical=.3)
                ax.add_artist(scalebar)

        t = np.arange(sta.shape[-1])*frame_duration*1000
        plt.subplots_adjust(wspace=0.3, hspace=0)
        ax = plt.subplot(rows, 1, 2)
        plt.plot(t, sta[max_i[0], max_i[1], :], label='center px')
        plt.plot(t, t1, label='Temporal 1')
        plt.plot(t, t2, label='Temporal 2')
        plt.xlabel('Time[ms]')
        plf.spineless(ax, 'trlb')  # Turn off spines using custom function

        plotpath = os.path.join(exp_dir, 'data_analysis', stimname, savefolder)
        if not os.path.isdir(plotpath):
            os.makedirs(plotpath, exist_ok=True)
        plt.savefig(os.path.join(plotpath, clusterids[i]+'.svg'), dpi=300)
        plt.close()
    print(f'Plotted checkerflicker SVD for {stimname}')

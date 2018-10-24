#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 11:51:38 2017

@author: ycan
"""
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import numpy as np
import os
import warnings

from . import util as plf
from ..calc import gaussfitter as gfit
from ..modules import analysisfuncs as asc
from ..modules import iofuncs as iof
from ..modules import miscfuncs as mf


def checkersurround(exp_name, stim_nr, filename=None, spikecutoff=1000,
                    ratingcutoff=4, staqualcutoff=0, inner_b=2,
                    outer_b=4):

    """
    Divides into center and surround by fitting 2D Gaussian, and plot
    temporal components.

    spikecutoff:
        Minimum number of spikes to include.

    ratingcutoff:
        Minimum spike sorting rating to include.

    staqualcutoff:
        Minimum STA quality (as measured by z-score) to include.

    inner_b:
        Defined limit between receptive field center and surround
        in units of sigma.

    outer_b:
        Defined limit of the end of receptive field surround.
    """

    exp_dir = iof.exp_dir_fixer(exp_name)
    stim_nr = str(stim_nr)
    if filename:
        filename = str(filename)

    if not filename:
        savefolder = 'surroundplots'
        label = ''
    else:
        label = filename.strip('.npz')
        savefolder = 'surroundplots_' + label

    _, metadata = asc.read_spikesheet(exp_name)
    px_size = metadata['pixel_size(um)']

    data = iof.load(exp_name, stim_nr, fname=filename)

    clusters = data['clusters']
    stas = data['stas']
    stx_h = data['stx_h']
    exp_name = data['exp_name']
    stimname = data['stimname']
    max_inds = data['max_inds']
    frame_duration = data['frame_duration']
    filter_length = data['filter_length']
    quals = data['quals'][-1, :]

    spikenrs = data['spikenrs']

    c1 = np.where(spikenrs > spikecutoff)[0]
    c2 = np.where(clusters[:, 2] <= ratingcutoff)[0]
    c3 = np.where(quals > staqualcutoff)[0]

    choose = [i for i in range(clusters.shape[0]) if ((i in c1) and
                                                      (i in c2) and
                                                      (i in c3))]
    clusters = clusters[choose]
    stas = list(np.array(stas)[choose])
    max_inds = list(np.array(max_inds)[choose])

    clusterids = plf.clusters_to_ids(clusters)

    t = np.arange(filter_length)*frame_duration*1000

    # Determine frame size so that the total frame covers
    # an area large enough i.e. 2*700um
    f_size = int(700/(stx_h*px_size))

    del data

    for i in range(clusters.shape[0]):

        sta_original = stas[i]
        max_i_original = max_inds[i]

        try:
            sta, max_i = mf.cut_around_center(sta_original,
                                              max_i_original, f_size)
        except ValueError:
            continue

        fit_frame = sta[:, :, max_i[2]]

        if np.max(fit_frame) != np.max(np.abs(fit_frame)):
            onoroff = -1
        else:
            onoroff = 1



        Y, X = np.meshgrid(np.arange(fit_frame.shape[1]),
                           np.arange(fit_frame.shape[0]))

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore',
                                    '.*divide by zero*.', RuntimeWarning)
            pars = gfit.gaussfit(fit_frame*onoroff)
            f = gfit.twodgaussian(pars)
            Z = f(X, Y)

        # Correcting for Mahalonobis dist.
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore',
                                    '.*divide by zero*.', RuntimeWarning)
            Zm = np.log((Z-pars[0])/pars[1])
        Zm[np.isinf(Zm)] = np.nan
        Zm = np.sqrt(Zm*-2)

        ax = plt.subplot(1, 2, 1)

        plf.stashow(fit_frame, ax)
        ax.set_aspect('equal')

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning)
            warnings.filterwarnings('ignore', '.*invalid value encountered*.')
            ax.contour(Y, X, Zm, [inner_b, outer_b],
                       cmap=plf.RFcolormap(('C0', 'C1')))

        barsize = 100/(stx_h*px_size)
        scalebar = AnchoredSizeBar(ax.transData,
                                   barsize, '100 µm',
                                   'lower left',
                                   pad=1,
                                   color='k',
                                   frameon=False,
                                   size_vertical=.2)
        ax.add_artist(scalebar)

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore',
                                    '.*invalid value encountered in*.',
                                    RuntimeWarning)
            center_mask = np.logical_not(Zm < inner_b)
            center_mask_3d = np.broadcast_arrays(sta,
                                                 center_mask[..., None])[1]
            surround_mask = np.logical_not(np.logical_and(Zm > inner_b,
                                                          Zm < outer_b))
            surround_mask_3d = np.broadcast_arrays(sta,
                                                   surround_mask[..., None])[1]

        sta_center = np.ma.array(sta, mask=center_mask_3d)
        sta_surround = np.ma.array(sta, mask=surround_mask_3d)

        sta_center_temporal = np.mean(sta_center, axis=(0, 1))
        sta_surround_temporal = np.mean(sta_surround, axis=(0, 1))

        ax1 = plt.subplot(1, 2, 2)
        l1 = ax1.plot(t, sta_center_temporal,
                      label='Center\n(<{}σ)'.format(inner_b),
                      color='C0')
        sct_max = np.max(np.abs(sta_center_temporal))
        ax1.set_ylim(-sct_max, sct_max)
        ax2 = ax1.twinx()
        l2 = ax2.plot(t, sta_surround_temporal,
                      label='Surround\n({}σ<x<{}σ)'.format(inner_b, outer_b),
                      color='C1')
        sst_max = np.max(np.abs(sta_surround_temporal))
        ax2.set_ylim(-sst_max, sst_max)
        plf.spineless(ax1)
        plf.spineless(ax2)
        ax1.tick_params('y', colors='C0')
        ax2.tick_params('y', colors='C1')
        plt.xlabel('Time[ms]')
        plt.axhline(0, linestyle='dashed', linewidth=1)

        lines = l1+l2
        labels = [line.get_label() for line in lines]
        plt.legend(lines, labels, fontsize=7)
        plt.title('Temporal components')
        plt.suptitle(f'{exp_name}\n{stimname}\n{clusterids[i]}')

        plt.subplots_adjust(wspace=.5, top=.85)

        plotpath = os.path.join(exp_dir, 'data_analysis',
                                stimname, savefolder)
        if not os.path.isdir(plotpath):
            os.makedirs(plotpath, exist_ok=True)

        plt.savefig(os.path.join(plotpath, clusterids[i])+'.svg',
                    format='svg', dpi=300)
        plt.close()
    print(f'Plotted checkerflicker surround for {stimname}')

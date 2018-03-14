#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 15:49:34 2018

@author: ycan
"""

import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
import gaussfitter as gfit
import iofuncs as iof
import miscfuncs as msc
import plotfuncs as plf
import analysis_scripts as asc
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from matplotlib.patches import Rectangle

import scalebars
import texplot

fig = texplot.texfig(1.2)

spikecutoff=1000
ratingcutoff=4
staqualcutoff=0
inner_b=2
outer_b=4

exp_name = '20180207'
stim_nr = 11


exp_dir = iof.exp_dir_fixer(exp_name)
stim_nr = str(stim_nr)

savefolder = 'surroundplots'

_, metadata = asc.read_ods(exp_name)
px_size = metadata['pixel_size(um)']

data = iof.load(exp_name, stim_nr)

clusters = data['clusters']
stas = data['stas']
stx_h = data['stx_h']
exp_name = data['exp_name']
stimname = data['stimname']
max_inds = data['max_inds']
frame_duration = data['frame_duration']
filter_length = data['filter_length']
quals = data['quals'][-1, :]

spikenrs = np.array([a.sum() for a in data['all_spiketimes']])

choose = [33]
clusters = clusters[choose]
stas = list(np.array(stas)[choose])
max_inds = list(np.array(max_inds)[choose])

clusterids = plf.clusters_to_ids(clusters)

t = np.arange(filter_length)*frame_duration*1000

# Determine frame size so that the total frame covers
# an area large enough i.e. 2*700um
f_size = int(700/(stx_h*px_size))

del data

rows, columns = 2, 2


for i in range(clusters.shape[0]):

    sta_original = stas[i]
    max_i_original = max_inds[i]

    try:
        sta, max_i = msc.cut_around_center(sta_original,
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

    ax1 = plt.subplot(rows, columns, 1)
    plf.subplottext('A', ax1)

    vmax = np.abs(fit_frame).max()
    vmin = -vmax
    im = plf.stashow(fit_frame, ax1)
    ax1.set_aspect('equal')
    plf.spineless(ax1)
    ax1.set_xticks([])
    ax1.set_yticks([])

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=UserWarning)
        warnings.filterwarnings('ignore', '.*invalid value encountered*.')
        ax1.contour(Y, X, Zm, [inner_b, outer_b],
                   cmap=plf.RFcolormap(('C0', 'C1')))

    barsize = 100/(stx_h*px_size)
    scalebar = AnchoredSizeBar(ax1.transData,
                               barsize, r'100 $\upmu$m',
#                               barsize, r'100 µm',
                               'lower left',
                               pad=1,
                               color='k',
                               frameon=False,
                               size_vertical=.2)
    ax1.add_artist(scalebar)

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

    ax2 = plt.subplot(rows, columns, 2)
    plf.subplottext('B', ax2, x=-.25, y=1.1)
    l1 = ax2.plot(t, sta_center_temporal,
                  color='C0')
    sct_max = np.max(np.abs(sta_center_temporal))
    ax2.set_yticks([])

    l2 = ax2.plot(t, sta_surround_temporal,
                  color='C1')
    sst_max = np.max(np.abs(sta_surround_temporal))
    plf.spineless(ax2)

    ax2.set_xlabel('Time[ms]')
    ax2.axhline(0, color='k', alpha=.5, linestyle='dashed', linewidth=1)

    data = iof.load(exp_dir, int(stim_nr)+1)
    stripesta = np.array(data['stas'])[choose][0]
    stripemax = np.array(data['max_inds'])[choose][0]
    stx_w = stx_h
    frame_duration = data['frame_duration']
    fits = np.array(data['fits'])[choose]
    onoroff = data['polarities'][choose]


    cut_time = int(100/(frame_duration*1000)/2)
    fsize_original = int(700/(stx_w*px_size))
    fsize = int(400/(stx_w*px_size))
    fsize_diff = fsize_original - fsize
    t = np.arange(filter_length)*frame_duration*1000
    vscale = fsize * stx_w*px_size

    stripesta, stripemax_i = msc.cutstripe(stripesta, stripemax, fsize*2)

    fitv = np.mean(stripesta[:, stripemax[1]-cut_time:stripemax[1]+cut_time+1],
               axis=1)

    s = np.arange(fitv.shape[0])

    ax3 = plt.subplot(rows, columns, 3)
    plf.stashow(stripesta, ax3)
    plf.subplottext('C', ax3)

    ax4 = plt.subplot(rows, columns, 4)
    ax4.plot(onoroff*fitv, -s, color='C2')
    plf.subplottext('D', ax4, x=-.25, y=1.1)
    plf.spineless(ax4)
    ax4.axvline(0, color='k', alpha=.5, linestyle='dashed', linewidth=1)

    ax4.set_axis_off()

    # Add scalebar
    time_set = 100 # milliseconds
    dist_set = 100 # micrometers

    barsize_time = time_set/(stx_h*px_size)
    barsize_distance = dist_set/(stx_h*px_size)

    scalebars.add_scalebar(ax3,
                           matchx=False, sizex=barsize_time,
                           labelx=f'{time_set} ms',
                           matchy=False, sizey=-barsize_distance,
#                           labely=f'{dist_set} µm',
                           labely=fr'{dist_set} $\upmu$m',
                           barwidth=1.2,
                           loc='lower right',
                           sep=2,
                           pad=0)
    # Add a box to show spatial component calculation
    ax3.add_patch(Rectangle(
                 (stripemax_i[-1]-cut_time, -0.375),
                 cut_time*2, stripesta.shape[0]-0.25,
                 linewidth=.75, linestyle='dashed',
                 edgecolor='C2', facecolor='none'))


    plt.subplots_adjust(wspace=.3, hspace=.35)
    texplot.savefig('comparesurroundcalc')
    plt.show()
    plt.close()
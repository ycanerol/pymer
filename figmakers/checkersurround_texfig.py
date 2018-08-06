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
import miscfuncs as mf
import plotfuncs as plf
import analysis_scripts as asc
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
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

_, metadata = asc.read_spikesheet(exp_name)
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

#c1 = np.where(spikenrs > spikecutoff)[0]
#c2 = np.where(clusters[:, 2] <= ratingcutoff)[0]
#c3 = np.where(quals > staqualcutoff)[0]
#
#choose = [i for i in range(clusters.shape[0]) if ((i in c1) and
#                                                  (i in c2) and
#                                                  (i in c3))]

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
    plf.subplottext('A', ax, x=0, y=1.3)

    vmax = np.abs(fit_frame).max()
    vmin = -vmax
    im = plf.stashow(fit_frame, ax, cmap=texplot.cmap)
    ax.set_aspect('equal')
    plf.spineless(ax)
    ax.set_axis_off()

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=UserWarning)
        warnings.filterwarnings('ignore', '.*invalid value encountered*.')
        ax.contour(Y, X, Zm, [inner_b, outer_b],
                   cmap=plf.RFcolormap(('C0', 'C1')))

    barsize = 100/(stx_h*px_size)
    scalebar = AnchoredSizeBar(ax.transData,
                               barsize, r'100 $\upmu$m',
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
    plf.subplottext('B', ax1, x=-.25, y=1.023)
    l1 = ax1.plot(t, sta_center_temporal,
#                  label=r'Center\n(<{}$\upsigma$)'.format(inner_b),
                  color='C0')
    sct_max = np.max(np.abs(sta_center_temporal))
#    ax1.set_ylim(-sct_max, sct_max)
#    ax2 = ax1.twinx()

    l2 = ax1.plot(t, sta_surround_temporal,
#                  label=r'Surround\n({}$\upsigma$<x<{}$\upsigma$)'.format(inner_b, outer_b),
                  color='C1')
    sst_max = np.max(np.abs(sta_surround_temporal))
#    ax2.set_ylim(-sst_max, sst_max)
    plf.spineless(ax1)
#    plf.spineless(ax2)
#    ax1.tick_params('y', colors='C0')
#    ax2.tick_params('y', colors='C1')
    plt.xlabel('Time[ms]')
    plt.axhline(0, color='k', alpha=.5, linestyle='dashed', linewidth=1)

#    lines = l1+l2
#    labels = [line.get_label() for line in lines]
#    plt.legend(lines, labels, fontsize=7)
#    plt.title('Temporal components')
#    plt.suptitle(f'{exp_name}\n{stimname}\n{clusterids[i]}')

    plt.subplots_adjust(wspace=.5, top=.85)

#    texplot.savefig('checkersurround')

    plt.show()
    plt.close()

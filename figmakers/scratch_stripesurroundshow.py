#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 12:08:14 2018

@author: ycan

Used for checking weird cells, plotting mesopic and photopic together.
"""


#import texplot
import iofuncs as iof
import analysis_scripts as asc
import miscfuncs as msc
import plotfuncs as plf

import matplotlib.pyplot as plt
import numpy as np

from stripesurround import onedgauss


#toshow = [
#        ('20180124', '04001', 'offweird1'),
#        ('20180124', '07401', 'offweird2'),
#        ('20180207', '02001', 'offweird3'),
#        ('20180207', '03002', 'offweird4'),
#        ]

data = np.load('/home/ycan/Documents/thesis/analysis_auxillary_files/'
               'thesis_csiplotting.npz')
cells = data['cells']
colors = data['colors']
csi_alts = np.ones(data['csi'].shape)*-1

rows = 2
columns = 2

#for i, (exp_name, clustertoplot, label) in enumerate(toshow):
for i, (exp_name, clustertoplot) in enumerate(cells):
#    break
    if '20180124' in exp_name or '20180207' in exp_name:
        stripeflicker = [6, 12]
    elif '20180118' in exp_name:
        stripeflicker = [7, 14]

#    fig = texplot.texfig(1, aspect=.8)
    fig = plt.figure(figsize=(6, 6))
    axes = [fig.add_subplot(rows, columns, i+1) for i in range(rows*columns)]

    for j, stimnr in enumerate(stripeflicker):

        exp_dir = iof.exp_dir_fixer(exp_name)

        _, metadata = asc.read_spikesheet(exp_dir)
        px_size = metadata['pixel_size(um)']

        data = iof.load(exp_name, stimnr)

        clusters = data['clusters']
        stas = data['stas']
        max_inds = data['max_inds']
        filter_length = data['filter_length']
        stx_w = data['stx_w']
        exp_name = data['exp_name']
        stimname = data['stimname']
        frame_duration = data['frame_duration']
        quals = data['quals']
        all_parameters = data['all_parameters']
        fits = data['fits']

        clusterids = plf.clusters_to_ids(clusters)
        index = np.where(np.array(clusterids) == clustertoplot)[0]
        index = np.asscalar(index)

        sta = data['stas'][index]
        max_i = data['max_inds'][index]
        onoroff = data['polarities'][index]
        csi = data['cs_inds'][index]
        fit = fits[index]
        popt = all_parameters[index]

        cut_time = int(100/(frame_duration*1000)/2)
        # Changed width from 700 micrometer to 400 to zoom in on the
        # region of interest. This shifts where the fit is drawn,
        # it's fixed when plotting.
        fsize_original = int(700/(stx_w*px_size))
        fsize = int(400/(stx_w*px_size))
        fsize_diff = fsize_original - fsize
        t = np.arange(filter_length)*frame_duration*1000
        vscale = fsize * stx_w*px_size

        sta, max_i = msc.cutstripe(sta, max_i, fsize*2)

        ax1 = axes[2*j]
        plf.subplottext(['A', 'C'][j], ax1, x=-.4)
        plf.subplottext(['Mesopic', 'Photopic'][j],
                        ax1, x=-.5, y=.5, rotation=90, va='center')
        plf.stashow(sta, ax1, extent=[0, t[-1], -vscale, vscale])
        ax1.set_xlabel('Time [ms]')
#        ax1.set_ylabel(r'Distance [$\upmu$m]')
        ax1.set_ylabel(r'Distance [μm]')

        fitv = np.mean(sta[:, max_i[1]-cut_time:max_i[1]+cut_time+1],
                       axis=1)

        s = np.arange(fitv.shape[0])

        ax2 = axes[2*j+1]
        plf.subplottext(['B', 'D'][j], ax2, x=-.1)
        plf.subplottext(f'Center-Surround Index: {csi:4.2f}',
                        ax2, x=.95, y=.15, fontsize=8, fontweight='normal')
        plf.spineless(ax2)
        ax2.set_yticks([])
        ax2.set_xticks([])
        ax2.plot(onoroff*fitv, -s, label='Data')
        # Displace the center of both distributions according to the difference
        fit = fit*onoroff
        surround_alt = np.abs(fit[fit<0].sum())
        center_alt = np.abs(fit[fit>0].sum())
        csi_alt = np.abs(fit[fit<0].sum())/(fit[fit>0].sum())
        csi_alts[j, i] = csi_alt
        print(surround_alt, center_alt, csi_alt)
        fit_cut = fit[2*fsize_diff:-2*fsize_diff]
        plf.subplottext(f'csi_alt: {csi_alt:4.2f}',
                ax2, x=.95, y=.05, fontsize=8, fontweight='normal')
        plf.subplottext(['', '■'][j], ax2, x=.50, y=.05, color=colors[i],
                        fontsize=8)
        popt = popt - np.array([0, 1, 0, 0, 1, 0])*fsize_diff*2
        ax2.plot(onedgauss(s, *popt[:3]), -s,  '--', label='Center')
        ax2.plot(-onedgauss(s, *popt[3:]), -s,  '--', label='Surround')
        ax2.plot(fit_cut, -s)

    plt.subplots_adjust(hspace = .3, wspace=.2)
    plt.suptitle(f'{i:>5d} {exp_name} {clustertoplot}')
#    texplot.savefig(label)
    plt.show()
#%%

ax = plt.subplot(111)
ax.scatter(csi_alts[0, :], csi_alts[1, :], c=colors, s=20, alpha=.8)
ax.plot([0, csi_alts.max()], [0, csi_alts.max()])
ax.set_aspect('equal')
#axis_lims = [-.03, 1.5]
axis_lims = None
ax.set_xlim(axis_lims)
ax.set_ylim(axis_lims)
plt.show()

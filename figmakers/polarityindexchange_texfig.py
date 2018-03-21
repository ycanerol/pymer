#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 08:50:54 2018

@author: ycan
"""
import numpy as np
import matplotlib.pyplot as plt
import iofuncs as iof
import plotfuncs as plf
import texplot


import scalebars

texplot.texfig(1.3, 1)


data = np.load('/home/ycan/Documents/thesis/'
               'analysis_auxillary_files/thesis_csiplotting.npz')

cells = data['cells']
groups = data['groups']
bias = data['bias']
csi = data['csi']
colorcategories = data['colorcategories']
colorlabels = data['colorlabels']


toplot = [['20180124', '02001'], # Increasing bias
          ['20180207', '03503'], # Decreasing bias
          ]

for j, (exp_name, clustertoplot) in enumerate(toplot):
    if '20180124' in exp_name or '20180207' in exp_name:
        onoffs = [3, 8]
    elif '20180118' in exp_name:
        onoffs = [3, 10]

    for i, (cond, stim) in enumerate(zip(['M', 'P'], onoffs)):
        expdata = iof.load(exp_name, stim)
        clusters = expdata['clusters']
        preframedur = expdata['preframe_duration']
        stimdur = expdata['stim_duration']
        clusterids = plf.clusters_to_ids(clusters)
        index = [i for i, cl in enumerate(clusterids) if cl == clustertoplot][0]

        fr = expdata['all_frs'][index]
        t = expdata['t']
        baselines = expdata['baselines'][index]

        plotind = [1, 3, 5, 7][i+2*j]
        ax = plt.subplot(4, 2, plotind)
        ax.plot(t, fr, 'k', linewidth=.5)
        plf.spineless(ax)
        if cond == 'M':
            plf.drawonoff(ax, preframedur, stimdur, h=.1)
            plf.subplottext(['A', 'B'][j], ax, x=-0.1)
        elif cond != 'M' and j == 0:
            scalebars.add_scalebar(ax,
                       matchx=False, sizex=.5,
                       labelx='500 ms',
                       matchy=False, sizey=30,
                       labely='30 Hz',
#                       labely=fr'{dist_set} $\upmu$m',
                       hidey=False,
                       barwidth=1.2,
                       loc='upper right',
                       sep=2,
                       pad=0)
        for pos in [2, 3, 5, 6]:
            ax.axvline(pos, color='k', alpha=.5, linestyle='dashed',
                       linewidth=1)
        ax.set_ylabel(cond)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylim(0, 130)
        ax.set_xlim(0, t.max())

ax3 = plt.subplot(2, 2, 2)
for color, group in zip(colorcategories, groups):
    ax3.plot(bias[:, group], color=color, linewidth=.4)
#plt.axis('equal')
plf.subplottext('C', ax3, x=-0.2, y=1.025)
ax3.set_xticks([0, 1])
ax3.set_xticklabels(['Mesopic', 'Photopic'])
ax3.set_ylabel('Polarity Index')
plf.spineless(ax3)

distrib = [len(group)/cells.shape[0] for group in groups]
ax4 = plt.subplot(2, 2, 4)
_, texts = ax4.pie(distrib, labels=colorlabels, colors=colorcategories,
#                     radius=.6,
                     startangle=-60,
                     labeldistance=1.15)
# Slightly shift the ON-OFF label to avoid overlap
[txt.set_horizontalalignment('right') for txt in texts[2:]]
plf.subplottext('D', ax4, x=-0.2, y=1.15)


plt.subplots_adjust(hspace=.4, wspace=.4)
texplot.savefig('polarityindexchange')
plt.show()

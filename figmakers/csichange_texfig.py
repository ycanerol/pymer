#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 10:42:18 2018

@author: ycan
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import iofuncs as iof
import plotfuncs as plf
import analysis_scripts as asc
import texplot


def csindexchange(exp_name, onoffcutoff=.5, qualcutoff=9):
    """
    Returns in center surround indexes and ON-OFF classfication in
    mesopic and photopic light levels.
    """
    # For now there are only three experiments with the
    # different light levels and the indices of stimuli
    # are different. To automate it will be tricky and
    # ROI is just not enough to justify; so they are
    # hard coded.
    if '20180124' in exp_name or '20180207' in exp_name:
        stripeflicker = [6, 12]
        onoffs = [3, 8]
    elif '20180118' in exp_name:
        stripeflicker = [7, 14]
        onoffs = [3, 10]

    exp_dir = iof.exp_dir_fixer(exp_name)
    exp_name = os.path.split(exp_dir)[-1]
    clusternr = asc.read_ods(exp_name)[0].shape[0]

    # Collect all CS indices, on-off indices and quality scores
    csinds = np.zeros((2, clusternr))
    quals = np.zeros((2, clusternr))

    onoffinds = np.zeros((2, clusternr))
    for i, stim in enumerate(onoffs):
        onoffinds[i, :] = iof.load(exp_name, stim)['onoffbias']

    for i, stim in enumerate(stripeflicker):
        data = iof.load(exp_name, stim)
        quals[i, :] = data['quals']
        csinds[i, :] = data['cs_inds']

    csinds_f = np.copy(csinds)
    quals_f = np.copy(quals)
    onoffbias_f = np.copy(onoffinds)

    # Filter them according to the quality cutoff value
    # and set excluded ones to NaN
    for j in range(quals.shape[1]):
        if not np.all(quals[:, j] > qualcutoff):
            quals_f[:, j] = np.nan
            csinds_f[:, j] = np.nan
            onoffbias_f[:, j] = np.nan

    # Define the color for each point depending on each cell's ON-OFF index
    # by appending the color name in an array.
    colors = []
    for j in range(onoffbias_f.shape[1]):
        if np.all(onoffbias_f[:, j] > onoffcutoff):
            # If it stays ON througout
            colors.append('blue')
        elif np.all(onoffbias_f[:, j] < -onoffcutoff):
            # If it stays OFF throughout
            colors.append('red')
        elif (np.all(onoffcutoff > onoffbias_f[:, j]) and
              np.all(onoffbias_f[:, j] > -onoffcutoff)):
            # If it's ON-OFF throughout
            colors.append('black')
        else:
            colors.append('orange')

    return csinds_f, colors, onoffbias_f, quals_f


def allinds(**kwargs):
    csi = np.empty([2, 0])
    colors = []
    bias = np.empty([2, 0])
    quals = np.empty([2, 0])
    cells = []
    for exp in ['20180118', '20180124', '20180207']:
        clusterids = plf.clusters_to_ids(asc.read_ods(exp)[0])
        cells.extend([(exp, cl_id) for cl_id in clusterids])
        csi_r, colors_r, bias_r, quals_r = csindexchange(exp, **kwargs)
        csi = np.hstack((csi, csi_r))
        colors.extend(colors_r)
        bias = np.hstack((bias, bias_r))
        quals = np.hstack((quals, quals_r))
    return csi, colors, bias, quals, cells

csi, colors, bias, quals, cells = allinds()
x = [np.nanmin(csi), np.nanmax(csi)]

scatterkwargs = {'c':colors, 'alpha':.6, 'linewidths':0, 's':4}

colorcategories = ['blue', 'red', 'black', 'orange']
colorlabels = ['ON', 'OFF', 'ON-OFF', 'Unstable']

# Create an array for all the colors to use with plt.legend()
patches = []
for color, label in zip(colorcategories, colorlabels):
    patches.append(mpatches.Patch(color=color, label=label))

fig = texplot.texfig(1.2)

ax = fig.add_subplot(111)
ax.scatter(csi[0, :], csi[1, :], **scatterkwargs)
ax.plot(x, x, 'r--', alpha=.5)
ax.legend(handles=patches, fontsize='xx-small')
#ax.set_xlim([0, .4])
#ax.set_ylim([0, .4])
ax.set_xlabel('Mesopic')
ax.set_ylabel('Photopic')
ax.set_title('Center Surround Index Change')
ax.set_aspect('equal')

texplot.savefig('csichange')
plt.show()
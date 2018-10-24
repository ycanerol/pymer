#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 01:35:23 2018

@author: ycan
"""

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import os

from .modules import iofuncs as iof
from .modules import plotfuncs as plf
from .modules import analysisfuncs as asc


def csindexchange(exp_name, onoffcutoff=.5, qualcutoff=9):
    """
    Plots the change in center surround indexes in different light
    levels. Also classifies based on ON-OFF index from the onoffsteps
    stimulus at the matching light level.
    """

    # For now there are only three experiments with the
    # different light levels and the indices of stimuli
    # are different. To automate it will be tricky and
    # ROI is just not enough to justify; so they are
    # hard coded.
    if '20180124' in exp_name or '20180207' in exp_name:
        stripeflicker = [6, 12, 17]
        onoffs = [3, 8, 14]
    elif '20180118' in exp_name:
        stripeflicker = [7, 14, 19]
        onoffs = [3, 10, 16]

    exp_dir = iof.exp_dir_fixer(exp_name)
    exp_name = os.path.split(exp_dir)[-1]
    clusternr = asc.read_spikesheet(exp_name)[0].shape[0]

    # Collect all CS indices, on-off indices and quality scores
    csinds = np.zeros((3, clusternr))
    quals = np.zeros((3, clusternr))

    onoffinds = np.zeros((3, clusternr))
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
            colors.append('white')

    scatterkwargs = {'c': colors, 'alpha': .6, 'linewidths': 0}

    colorcategories = ['blue', 'red', 'black']
    colorlabels = ['ON', 'OFF', 'ON-OFF']

    # Create an array for all the colors to use with plt.legend()
    patches = []
    for color, label in zip(colorcategories, colorlabels):
        patches.append(mpatches.Patch(color=color, label=label))

    x = [np.nanmin(csinds_f), np.nanmax(csinds_f)]

    plt.figure(figsize=(12, 6))
    ax1 = plt.subplot(121)
    plt.legend(handles=patches, fontsize='small')
    plt.scatter(csinds_f[0, :], csinds_f[1, :], **scatterkwargs)
    plt.plot(x, x, 'r--', alpha=.5)
    plt.xlabel('Low 1')
    plt.ylabel('High')

    ax1.set_aspect('equal')
    plf.spineless(ax1)

    ax2 = plt.subplot(122)
    plt.scatter(csinds_f[0, :], csinds_f[2, :], **scatterkwargs)
    plt.plot(x, x, 'r--', alpha=.5)
    plt.xlabel('Low 1')
    plt.ylabel('Low 2')
    ax2.set_aspect('equal')
    plf.spineless(ax2)

    plt.suptitle(f'Center-Surround Index Change\n{exp_name}')
    plt.text(.8, -0.1, f'qualcutoff:{qualcutoff} onoffcutoff:{onoffcutoff}',
             fontsize='small', transform=ax2.transAxes)
    plotsave = os.path.join(exp_dir, 'data_analysis', 'csinds')
    plt.savefig(plotsave+'.svg', format='svg', bbox_inches='tight')
    plt.savefig(plotsave+'.pdf', format='pdf', bbox_inches='tight')
    plt.show()
    plt.close()

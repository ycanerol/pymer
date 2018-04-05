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
from scipy import stats
import texplot

qualcutoff = 11
def csindexchange(exp_name, onoffcutoff=.5, qualcutoff=qualcutoff):
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

    # Calculate the change of polarity for each cell
    # np.diff gives the high-low value
    biaschange = np.diff(onoffbias_f, axis=0)[0]

    # Define the color for each point depending on each cell's ON-OFF index
    # by appending the color name in an array.
    colors = []
    for j in range(onoffbias_f.shape[1]):
        if np.all(onoffbias_f[:, j] > onoffcutoff):
            # If it stays ON througout
            colors.append(colorcategories[0])
        elif np.all(onoffbias_f[:, j] < -onoffcutoff):
            # If it stays OFF throughout
            colors.append(colorcategories[1])
        elif (np.all(onoffcutoff > onoffbias_f[:, j]) and
              np.all(onoffbias_f[:, j] > -onoffcutoff)):
            # If it's ON-OFF throughout
            colors.append(colorcategories[2])
        elif biaschange[j] > 0:
            # Increasing polarity
            # If it's not consistent in any category and
            # polarity change is positive
            colors.append(colorcategories[3])
        elif biaschange[j] < 0:
            # Decreasing polarity
            colors.append(colorcategories[4])
        else:
            colors.append('yellow')

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
#%%
colorcategories = ['mediumblue', 'crimson', 'darkorange', 'springgreen',
                   'deepskyblue']
colorlabels = ['ON', 'OFF', 'ON-OFF',
               'Increased polarity index', 'Decreased polarity index']

csi, colors, bias, quals, cells = allinds()

include = np.all(quals > qualcutoff, axis=0)
include = np.logical_and(include, ~np.any(np.isnan(bias), axis=0))
include = np.logical_and(include, ~np.any(np.isnan(csi), axis=0))
include = np.logical_and(include, texplot.exclude_cells(cells))

quals = quals[:, include]
csi = csi[:, include]
bias = bias[:, include]
colors = [color for i, color in enumerate(colors) if include[i]]
cells = [cell for i, cell in enumerate(cells) if include[i]]

x = [np.nanmin(csi), np.nanmax(csi)]
scatterkwargs = {'c':colors, 'alpha':.8, 'linewidths':.5,
                 'edgecolor':'k',
                 's':35}

# Create an array for all the colors to use with plt.legend()
patches = []
for color, label in zip(colorcategories, colorlabels):
    patches.append(mpatches.Patch(color=color, label=label))

fig = texplot.texfig(.93, aspect=1.85)

ax = plt.subplot2grid((5, 3), (0, 0), colspan=3, rowspan=3)
ax.plot(x, x, 'k--', alpha=.5, linewidth=.8)
plf.subplottext('A', ax, x=-0.05, y=1.03)
ax.scatter(csi[0, :], csi[1, :], **scatterkwargs)

# Mark the example cells with an asterisk
asterixes = [(0.03443051,  0.19385925), # Example ON cell 20180207 03001
         (0.03238909,  0.29553824)] # Example OFF cell 20180118 23102
for asterix in asterixes:
    ax.text(*asterix, '*', color='k')

ax.legend(handles=patches, fontsize='xx-small')
ax_lims = ax.get_xlim()

ax.set_xlabel('Center Surround Index at \\textbf{Mesopic} conditions')
ax.set_ylabel('Center Surround Index at \\textbf{Photopic} conditions')
ax.set_aspect('equal')

groups = []

for i, color in enumerate(colorcategories):
    group = [index for index, c in enumerate(colors) if c == color]
    groups.append(group)
    ax = plt.subplot2grid((5,3), (3+int((np.round((i-1)/3))), i%3))

    ax.set_xlim(ax_lims)
    ax.set_ylim(ax_lims)
    tick_locs = [0, 0.2, 0.4]
    ax.set_xticks(tick_locs)
    ax.set_yticks(tick_locs)

    ax.scatter(csi[0, :], csi[1, :], s=6, c='grey')
    ax.scatter(csi[0, group], csi[1, group], s=6, c=color, linewidths=.5)
    ax.plot(x, x, 'k--', alpha=.5, linewidth=.5)

    plf.subplottext(['B', 'C', 'D', 'E', 'F'][i], ax, x=-.25)
    ax.set_aspect('equal')
    plf.spineless(ax, 'tr')
plt.subplots_adjust(hspace=.4, wspace=.4)
texplot.savefig('csichange')

plt.show()

# Perform non-parametric paired test to test significance
stat_test = stats.wilcoxon(csi[0, :], csi[1, :])
print(f'Wilcoxon signed-rank test p-val: {stat_test[1]:7.2e}')
for label, group in zip(colorlabels, groups):
    stat_test_grp = stats.wilcoxon(csi[0, group], csi[1, group])
    print(f'Wilcoxon p-val for {label:25s}: {stat_test_grp[1]:7.2e}')



np.savez('/home/ycan/Documents/thesis/analysis_auxillary_files/thesis_csiplotting.npz',
         cells=cells,
         bias=bias,
         include=include,
         colors=colors,
         groups=groups,
         stat_test=stat_test,
         colorcategories=colorcategories,
         colorlabels=colorlabels,
         csi=csi)

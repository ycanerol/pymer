#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 11:54:32 2017

@author: ycan

Perform singular value decomposition on STA to separate space and time
components.
The assumption here is that space and time are independent components.
"""


import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import scipy.ndimage as ndi
import os

try:
    import miscfuncs as mf
    import plotfuncs as plf
except ImportError:
    import sys
    sys.path.append('/home/ycan/repos/pymer/modules/')
    import miscfuncs as mf
    import plotfuncs as plf


def plotsvd(file, f_size=10, filter_size=1):

    data = np.load(file)
    filename = os.path.split(file)[-1].split('.')[0]

    sta = data['sta_unscaled']
    max_i = data['max_i']

    sta, max_i = mf.cut_around_center(sta, max_i, f_size=f_size)

    fit_frame = sta[:, :,  max_i[2]]

    # %%
    sp1, sp2, t1, t2, u, v = mf.svd(sta)

    sp1_filtered = ndi.filters.gaussian_filter(sp1,
                                               sigma=(filter_size,
                                                      filter_size))
    sp2_filtered = ndi.filters.gaussian_filter(sp2,
                                               sigma=(filter_size,
                                                      filter_size))
    ff_filtered = ndi.filters.gaussian_filter(fit_frame,
                                              sigma=(filter_size,
                                                     filter_size))

    plotthese = [fit_frame, sp1, sp2, ff_filtered, sp1_filtered, sp2_filtered]

    fig = plt.figure(dpi=130)
    plt.suptitle('{}\n frame size: {}'.format(filename, f_size))
    rows = 3
    columns = 3
    vmax = np.max(np.abs(sp1))
    vmin = -vmax

    for i in range(6):
        ax = plt.subplot(rows, columns, i+1)
        im = plt.imshow(plotthese[i], vmin=vmin, vmax=vmax, cmap=plf.RFcolormap())
        ax.set_aspect('equal')
        plt.xticks([])
        plt.yticks([])
        for child in ax.get_children():
            if isinstance(child, matplotlib.spines.Spine):
                child.set_color('C{}'.format(i % 3))
                child.set_linewidth(2)
        if i==0: plt.title('center px'); fig.colorbar(im)
        elif i==1: plt.title('SVD spatial 1')
        elif i==2: plt.title('SVD spatial 2')
        if i==0: plt.ylabel('Non-filtered')
        if i==3: plt.ylabel('Gaussian filtered')

    ax = plt.subplot(rows, 1, 3)
    plt.plot(sta[max_i[0], max_i[1], :], label='center px')
    plt.plot(t1, label='Temporal 1')
    plt.plot(t2, label='Temporal 2')
    plf.spineless(ax, 'trlb')  # Turn off spines using custom function
    return fig


# %%
fsizes = [0, 5, 10, 15, 20]
file = '/home/ycan/Documents/data/2017-08-02/analyzed/5_SP_C10602.npz'
for i in fsizes:
    filename = os.path.split(file)[-1].split('.')[0]
    plotsvd(file, f_size=i, filter_size=1)
    if False:
        plt.show()
    else:
        plt.savefig('/home/ycan/Documents/notes/'
                    '2017-11-15/svdplots/{}_{}.svd'.format(filename, i),
                    dpi=200,
                    bbox_inches='tight')

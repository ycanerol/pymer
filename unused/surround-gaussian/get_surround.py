#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 11:51:38 2017

@author: ycan
"""

import numpy as np
import matplotlib.pyplot as plt
import gaussfitter as gfit

try:
    import miscfuncs as mf
except ImportError:
    import sys
    sys.path.append('/home/ycan/repos/pymer/modules/')
    import miscfuncs as mf

files = mf.readexps(directory='/home/ycan/Documents/data/2017-08-02',
                    test=True)
#files = np.reshape(files[:, 25], (3,1))

for i in range(files.shape[1]):

    file_path = str(files[0, i])
    exp_date = str(files[1, i])
    cluster = str(files[2, i])
    data = np.load(file_path)

    sta_original = data['sta_unscaled']
    max_i_original = data['max_i']

    if data['total_spikes'] < 350:
        continue

    f_size = 10

    sta, max_i = mf.cut_around_center(sta_original, max_i_original, f_size)

    # Isolate the frame that wtll be used to fit the 2D Gaussian distribution
    # This is a first pass approach, just to see if anything comes out
    fit_frame = sta[:, :, max_i[2]]
#    show_sta(sta, max_i)

    # %% Concenctric rings using pixel distances

    plt.figure(figsize=(12, 10))
    plt.suptitle('{} {:>5}'.format(exp_date, cluster))
    ring_sizes1 = [0, 3]
    ring_sizes2 = [3, 7]

    for i in range(ring_sizes1[0], ring_sizes1[1]):
        plt.subplot(4, 2, 1)
        plt.title('Temporal filter at n pixel distance from center pixel')
        masked = mf.ringmask(sta, max_i, i)
        if not masked[1]:
            plt.plot(np.mean(masked[0], axis=(0, 1)), label=str(i))
        plt.legend()

    for i in range(ring_sizes2[0], ring_sizes2[1]):
        plt.subplot(4, 2, 3)
        masked = mf.ringmask(sta, max_i, i)
        if not masked[1]:
            plt.plot(np.mean(masked[0], axis=(0, 1)), label=str(i))
    plt.legend()

# %% Fit 2D Gaussian
#    fit_frame = sta[:, :, max_i[2]]
#    f_size = 20
#
#    if f_size is not 0:
#        fit_frame = fit_frame[max_i[0]-f_size:max_i[0]+f_size+1,
#                              max_i[1]-f_size:max_i[1]+f_size+1]

    pars = gfit.gaussfit(fit_frame)

    f = gfit.twodgaussian(pars)

    Y, X = np.meshgrid(np.arange(fit_frame.shape[1]), np.arange(fit_frame.shape[0]))
    Z = f(X, Y)
    # Correcting for Mahalonobis dist.
    # Using a second variable Zm2 to not break how it currently works and
    # easily revert
    Zm2 = np.log((Z-pars[0])/pars[1])
    Zm2[np.isinf(Zm2)] = np.nan
    Zm = np.sqrt(Zm2*-2)
    # To workaround negative values from before, remove minus from Zm comparisons to fix this
#    Zm = -Zm
#    Zmr = np.ceil(Zm)

    ax = plt.subplot(2, 2, 2)
    if np.max(fit_frame) != np.max(np.abs(fit_frame)):
        fit_frame = -fit_frame
    ax.imshow(fit_frame, cmap='BuGn')
    ax.contour(f(*np.indices(fit_frame.shape)), 4, cmap=plt.cm.Reds)
#    ax.contour(X, Y, -Zm)

    from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

    scalebar = AnchoredSizeBar(ax.transData,
                               3, '180 µm', 'lower left',
                               pad=1,
                               color='grey',
                               frameon=False,
                               size_vertical=.3)
    ax.add_artist(scalebar)
#    plt.show()
# %%
#    plt.figure(figsize=(12, 12))
#    for i in range(0, 9):
#        plt.subplot(3, 3, i+1)
#        plt.title('Distance {} SD'.format(i+1))
#        plt.imshow(np.logical_and(Zm < -i, Zm > -i-1))
#        plt.axis('off')
#    plt.show()

    # Boundaries between center and surround and limit of surround
    inner_b = 2
    outer_b = 4

    center_mask = np.logical_not(Zm < inner_b)
    center_mask_3d = np.broadcast_arrays(sta, center_mask[..., None])[1]
    surround_mask = np.logical_not(np.logical_and(Zm > inner_b, Zm < outer_b))
    surround_mask_3d = np.broadcast_arrays(sta, surround_mask[..., None])[1]

    # %%

    plt.subplot(2, 3, 4)
    plt.imshow(center_mask)
    plt.title('Center (<{}$\sigma$)'.format(inner_b))
    plt.subplot(2, 3, 5)
    plt.imshow(surround_mask)
    plt.title('Surround (Between {}$\sigma$ and {}$\sigma$)'.format(inner_b, outer_b))
#    plt.show()

    sta_center = np.ma.array(sta, mask=center_mask_3d)
    sta_surround = np.ma.array(sta, mask=surround_mask_3d)

    sta_center_temporal = np.mean(sta_center, axis=(0, 1))
    sta_surround_temporal = np.mean(sta_surround, axis=(0, 1))

#    f = plt.figure()
    ax = plt.subplot(2, 3, 6)
#    plt.plot(sta[max_i[0], max_i[1], :], label='Center pixel')
    plt.plot(sta_center_temporal, label='Center')
    plt.plot(sta_surround_temporal, label='Surround')
    plt.text(0.5, 0.2,
             'Correlation: {:5.3f}'.format(np.corrcoef(sta_center_temporal,
                                                       sta_surround_temporal)[0][1]),
             size=9, transform=ax.transAxes)
    plt.axhline(0, linestyle='dashed', linewidth=1)
    plt.legend()
    # Quickly turn plotting on or off
    if True:
        plt.show()
    else:
        plt.savefig('/home/ycan/Documents/notes/2017-11-01/'
                    'plots_mahalonobisd_corrected/{}-{:0>5}.svg'.format(exp_date, cluster),
                    format='svg', dpi=300)
    plt.close()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 11:51:38 2017

@author: ycan
"""

import numpy as np
import matplotlib.pyplot as plt
import glob
import gaussfitter as gfit


def show_sta(sta, max_i, f_size=10):
    plt.plot(figsize=(12, 12), dpi=200)
    sta_min = np.min(sta)
    sta_max = np.max(sta)

    for i in range(20):
        plt.subplot(4, 5, i+1)

        plt.imshow(sta[:, :, i], vmin=sta_min, vmax=sta_max , cmap='Greys')
        plt.axis('off')
    plt.show()


def readexps(test=False,
             directory='/home/ycan/Documents/'
                       'Yunus_rotation_2017_06/data/Experiments/Mouse/'):
    # TODO: stimulus_order needs to be parametrized/automated properly
    stimulus_order = 5
    file_paths = glob.glob(directory+'*/analyzed/{}*.npz'.format(stimulus_order))
    exp_names = [i.split('/')[-3] for i in file_paths]
    clusters = [i.split('/')[-1].split('C')[-1].split('.')[0] for i in file_paths]

    files = np.array([file_paths, exp_names, clusters])

    # Use only one file for testing
    if test:
        files = files[:, np.random.randint(files.shape[1])]
        files = files[:, np.newaxis]

    return files


def ringmask(data, center_px, r):
    # Create a ring shaped mask  in spatial dimention
    # and apply to the STA along time axis

    # HINT: "True" masks the value out
    mask = np.array([True]*data.size).reshape(data.shape)

    cx, cy, _ = center_px

    # Check if the specified ring size is larger than the shape
    outofbounds = (cx+r > data.shape[0]-1 or cx-r < 0 or
                   cx+r > data.shape[0]-1 or cx-r < 0)

    mask[cx-r:cx+r+1, cy-r:cy+r+1, :] = False
    mask[cx-(r-1):cx+(r), cy-(r-1):cy+(r), :] = True

    masked_data = np.ma.array(data, mask=mask)

    if outofbounds: masked_data = None

    return masked_data, outofbounds


def cut_around_center(sta_original, max_i_o, f_size):
    if f_size is not 0:
        sta = sta_original[max_i_o[0]-f_size:max_i_o[0]+f_size+1,
                           max_i_o[1]-f_size:max_i_o[1]+f_size+1,
                           :]
        max_i = np.append([f_size]*2, max_i_o[2])
    else:
        sta = sta_original
        max_i = max_i_o
    return sta, max_i


counter = 0
files = readexps(test=True, directory='/home/ycan/Documents/data/2017-08-02')

for i in range(files.shape[1]):

    file_path = str(files[0, i])
    exp_date = str(files[1, i])
    cluster = str(files[2, i])
    data = np.load(file_path)

    sta_original = data['sta_unscaled']
    max_i_original = data['max_i']

    f_size = 10

    sta, max_i = cut_around_center(sta_original, max_i_original, f_size)

    # Isolate the frame that wtll be used to fit the 2D Gaussian distribution
    # This is a first pass approach, just to see if anything comes out
    fit_frame = sta[:, :, max_i[2]]
    show_sta(sta, max_i)


    # %% Concenctric rings using pixel distances

    plt.figure(figsize=(12, 10))
    plt.suptitle('Temporal filter at n pixel distance from center pixel\n{} {:>5}'.format(exp_date, cluster))
    ring_sizes1 = [0, 3]
    ring_sizes2 = [3, 7]
    for i in range(ring_sizes1[0], ring_sizes1[1]):
        plt.subplot(2, 1, 1)
        masked = ringmask(sta, max_i, i)
        if not masked[1]:
            plt.plot(np.mean(masked[0], axis=(0, 1)), label=str(i))
        plt.legend()

    for i in range(ring_sizes2[0], ring_sizes2[1]):
        plt.subplot(2, 1, 2)
        masked = ringmask(sta, max_i, i)
        if not masked[1]:
            plt.plot(np.mean(masked[0], axis=(0, 1)), label=str(i))
    plt.legend()
#    plt.savefig('/home/ycan/Documents/notes/week2/plots/{}-{:0>5}.svg'.format(exp_date, cluster),
#                format='svg', dpi=300)
    plt.show()
    plt.close()
#    plt.show()


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
    Zm = np.log((Z-pars[0])/pars[1])
    Zmr = np.ceil(Zm)

    plt.imshow(fit_frame)
    plt.contour(f(*np.indices(fit_frame.shape)), 4, cmap=plt.cm.Blues)
    plt.show()
# %%
    plt.figure(figsize=(12, 12))
    for i in range(0, 9):
        plt.subplot(3, 3, i+1)
        plt.title('Mahalonobis distance: {}'.format(i+1))
        plt.imshow(np.logical_and(Zm < -i, Zm > -i-1))
        plt.axis('off')
    plt.show()

    center_mask = np.logical_not(Zm > -3)
    center_mask_3d = np.broadcast_arrays(sta, center_mask[..., None])[1]
    surround_mask = np.logical_not(np.logical_and(Zm < -3, Zm > -9))
    surround_mask_3d = np.broadcast_arrays(sta, surround_mask[..., None])[1]

    plt.subplot(121)
    plt.imshow(center_mask)
    plt.title('Center (<3$\sigma$)')
    plt.subplot(122)
    plt.imshow(surround_mask)
    plt.title('Surround (Between 3$\sigma$ and 9$\sigma$)')
    plt.show()

    sta_center = np.ma.array(sta, mask=center_mask_3d)
    sta_surround = np.ma.array(sta, mask=surround_mask_3d)

    sta_center_temporal = np.mean(sta_center, axis=(0, 1))
    sta_surround_temporal = np.mean(sta_surround, axis=(0, 1))

    f = plt.figure(); ax = plt.subplot(111)
#    plt.plot(sta[max_i[0], max_i[1], :], label='Center pixel')
    plt.plot(sta_center_temporal, label='Center')
    plt.plot(sta_surround_temporal, label='Surround')
    plt.text(0.8, 0.05,
             'Correlation: {:5.3f}'.format(np.corrcoef(sta_center_temporal, sta_surround_temporal)[0][1]),
             size=9, transform= ax.transAxes)
    plt.axhline(0, linestyle='dashed', linewidth = 1)
    plt.legend()
    plt.show()


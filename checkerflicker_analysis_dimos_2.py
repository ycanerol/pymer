#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 17:59:27 2017

@author: ycan
Analyze checkerflicker stimulus

Copied&'forked' on 2017-10-23 to generalize for other experiments
Original: checkerflicker_analysis_dimos.py

"""

import sys
import numpy as np
from collections import Counter
import scipy.io
import h5py
# Custom packages
import randpy
try:
    import lnp_checkerflicker as lnpc
    import lnp
except:
    sys.path.append('/home/ycan/Documents/scripts/modules')
    import lnp_checkerflicker as lnpc
    import lnp

main_dir = '/home/ycan/Documents/data/2017-02-10/'

cells_classified = main_dir+'clusters2.mat'

frames_path = 'frametimes/10_checkerflicker_2x2bw_4blinks_frametimings.mat'
stimulus_name = frames_path.split('/')[-1]
stimulus_order = stimulus_name.split('_')[0]

sx = 300
sy = 400

f = scipy.io.loadmat(cells_classified)

#%%

clusters = np.array(f.get('clusters')).astype(int)
ratings = np.array(f.get('ratings')).astype(int)

ratings = ratings.reshape(ratings.shape[0])
clusters = clusters[ratings < 4]
# %%

files = ['{}{:02}'.format(clusters[i, 0],
                          clusters[i, 1]) for i in range(clusters.shape[0])]
#files=['4105']

first_run_flag = True

for filename in files:
    if first_run_flag:
        f = scipy.io.loadmat(main_dir+frames_path)
        ftimes = (np.array(f.get('ftimes'))/1000)
        ftimes = ftimes.reshape((ftimes.size,))
        # Average difference between two frames in miliseconds
        dt = np.average(np.ediff1d(ftimes))
        del f

        total_frames = ftimes.shape[0]
        # In case the experiment is too long, take the first 100000 frames
        if total_frames>66000:
            print('Total frames is too long ({0}), '
                  'resetting'.format(total_frames))
            total_frames = 66000
        ftimes = ftimes[:total_frames]
        filter_length = 20  # Specified in nr of fra        # Generate enough random numbers, reshape and reorder them and map to
        # -1 and 1 for checkerflickermes

        # Generate enough random numbers, reshape and reorder them and map to
        # -1 and 1 for checkerflicker
#        rnd_numbers, seed = randpy.ran1(-10000, total_frames*sx*sy)
#        rnd_numbers = np.array(rnd_numbers).reshape(sx, sy,
#                                                    total_frames,
#                                                    order='F')
#        stimulus = np.array(np.where(rnd_numbers > .5, 1, -1), dtype='int8')
#        del rnd_numbers
        stimulus = loadchecker(sx=300, sy=400, chunks=11, clength=6000)
#        stimulus = np.load('/home/ycan/Documents/Yunus_rotation_2017_06/data/'
#                           'checkerflickerstimulus.npy')[:, :, :total_frames]

        first_run_flag = False

    spike_path = main_dir+'rasters/'+str(stimulus_order)+'_SP_C'+filename+'.txt'
    save_path = main_dir+'analyzed/'+str(stimulus_order)+'_SP_C'+'{:0>5}'.format(filename)

    spike_file = open(spike_path)
    spike_times = np.array([float(line) for line in spike_file])
    spike_file.close()

    spike_counts = Counter(np.digitize(spike_times, ftimes))
    spikes = np.array([spike_counts[i] for i in range(total_frames)])

    total_spikes = np.sum(spikes)
    if total_spikes < 2:
        continue

# %%
    sta_unscaled, max_i, temporal = lnpc.sta(spikes,
                                             stimulus,
                                             filter_length,
                                             total_frames)
    max_i = lnpc.check_max_i(sta_unscaled, max_i)

    stim_gaus = lnpc.stim_weighted(sta_unscaled, max_i, stimulus)

    sta_weighted, _ = lnp.sta(spikes, stim_gaus, filter_length, total_frames)

    w, v, _, _, _ = lnpc.stc(spikes, stim_gaus,
                             filter_length, total_frames, dt)

    bins = []
    spike_counts_in_bins = []
    for i in [sta_weighted, v[:, 0]]:
        a, b = lnpc.nlt_recovery(spikes, stim_gaus, i, 60, dt)
        bins.append(a)
        spike_counts_in_bins.append(b)

    sta_weighted, bins[0], \
    spike_counts_in_bins[0], \
    _, _ = lnpc.onoffindex(sta_weighted, bins[0],
                           spike_counts_in_bins[0])

    v[:, 0], bins[1],\
    spike_counts_in_bins[1],\
    peak, onoffindex = lnpc.onoffindex(v[:, 0], bins[1],
                                       spike_counts_in_bins[1])
    np.savez(save_path,
             sta_unscaled=sta_unscaled,
             sta_weighted=sta_weighted,
             stimulus_order=stimulus_order,
             total_frames=total_frames,
             temporal=temporal,
             v=v,
             w=w,
             sx=sx,
             sy=sy,
             max_i=max_i,
             spike_path=spike_path,
             filename=filename,
             bins=bins,
             spike_counts_in_bins=spike_counts_in_bins,
             onoffindex=onoffindex,
             total_spikes=total_spikes,
             peak=peak,
             stimulus_name=stimulus_name
             )

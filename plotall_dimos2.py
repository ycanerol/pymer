#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 12:16:38 2017

@author: ycan

Plotting for data from Dimos

The stimulus names are different than Fernando's data (Salamander).
2 is checkerflicker
3 is full field flicker

"""
import os
import numpy as np
import matplotlib.pyplot as plt

main_dir = '/home/ycan/Documents/data/2017-02-10/analyzed/'

exp_name = main_dir.split('/')[-3]+'_'+main_dir.split('/')[-2]

allfiles = os.listdir(main_dir)

stimulus_order='10'

files_f = []  # Full field flicker
files_c = []  # Checkerflicker

for i in allfiles:
    if i[-4:] == '.npz':
        if i.split('_')[0] == str(2): pass
#            files_f.append(i.split('C')[-1].split('.')[0])
        elif i.split('_')[0] == str(stimulus_order):
            files_c.append(i.split('C')[-1].split('.')[0])

#files = [i for i in files_c if i in files_f]
files_c = files_c[3:]

for i in files_c:
#    # Changed this part because of stimulus order difference
    fname_c = main_dir+stimulus_order+'_SP_C'+i+'.npz'
#    fname_f = main_dir+'3_SP_C'+i+'.npz'

#    f = np.load(fname_f)
    c = np.load(fname_c)

    savepath = '/'.join(main_dir.split('/')[:-1])+'/SP_C'+i


    # %% plot all
#    plt.figure(figsize=(12, 12), dpi=200)
#    plt.suptitle([' '.join(str(c['spike_path'])
#                 .split('rasters')[0].split('Experiments')[1]
#                 .split('/'))+str(i)])
#    plt.subplot(3, 3, 1)
#    plt.plot(f['sta'])
#    plt.plot(f['v'][:, 0])
#    plt.title('Filters')
#    plt.axvline(f['peak'], linewidth=1, color='r', linestyle='dashed')
#    plt.legend(['STA', 'Eigenvalue 0', 'Peak'], fontsize='small')
#    plt.xticks(np.linspace(0, 20, int(20/2+1)))
#    plt.ylabel('Full field flicker\n$\\regular_{Linear\,\,\,output}$',
#               fontsize=16)
#    plt.xlabel('Time')

#    ax = plt.subplot(3, 3, 2)
#    plt.plot(f['bins_sta'], f['spikecount_sta'], '-')
#    plt.plot(f['bins_stc'], f['spikecount_stc'], '-')
#    plt.text(.5, .99, 'On-Off Bias: {:2.2f}\nTotal spikes: {}'
#             .format(float(f['onoffindex']), f['total_spikes']),
#             horizontalalignment='center',
#             verticalalignment='top',
#             transform=ax.transAxes)
#    plt.title('Non-linearities')
#    plt.ylabel('Firing rate')
#    plt.xlabel('Linear output')

#    plt.subplot(3, 3, 3)
#    plt.plot(f['w'], 'o')
#    plt.title('Eigenvalues of covariance matrix')
#    plt.xticks(np.linspace(0, 20, int(20/2+1)))
#    plt.xlabel('Eigenvalue index')
#    plt.ylabel('Variance')

    plt.subplot(3, 3, 4)
    plt.plot(c['sta_weighted'])
    plt.plot(c['v'][:, 0])
    plt.plot(c['temporal'])
    plt.axvline(c['peak'], linewidth=1, color='r', linestyle='dashed')
    plt.title('Filters')
    plt.ylabel('Checkerflicker\n$\\regular_{Linear\,\,\,output}$', fontsize=16)
    plt.xlabel('Time')
    plt.xticks(np.linspace(0, 20, int(20/2+1)))
    plt.legend(['Weighted stimulus', 'Eigenvalue 0', 'Brightest pixel',
                'Peak'], fontsize='small')

    ax = plt.subplot(3, 3, 5)
    for i in range(len(c['bins'])):
        plt.plot(c['bins'][i], c['spike_counts_in_bins'][i], '-')
    plt.text(.5, .99, 'On-Off Bias: {:2.2f}\nTotal spikes: {}'
             .format(float(c['onoffindex']), c['total_spikes']),
             horizontalalignment='center',
             verticalalignment='top',
             transform=ax.transAxes)
    plt.title('Non-linearities')
    plt.xlabel('Linear output')
    plt.ylabel('Firing rate')

    plt.subplot(3, 3, 6)
    plt.plot(c['w'], 'o')
    plt.title('Eigenvalues of covariance matrix')
    plt.xticks(np.linspace(0, 20, int(20/2+1)))
    plt.xlabel('Eigenvalue index')
    plt.ylabel('Variance')

    plt.subplot(3, 3, 7)
    plt.imshow(c['sta_unscaled'][:, :, c['max_i'][2]].reshape((c['sx'],
               c['sy'],)),
               cmap='Greys',
               vmin=np.min(c['sta_unscaled']),
               vmax=np.max(c['sta_unscaled']))
    plt.title('Receptive field')

    plt.subplot(3, 3, 8)
    f_size = 50
    plt.imshow(c['sta_unscaled'][c['max_i'][0]-f_size:c['max_i'][0]+f_size+1,
                                 c['max_i'][1]-f_size:c['max_i'][1]+f_size+1,
                                 int(c['max_i'][2])],
               cmap='Greys',
               vmin=np.min(c['sta_unscaled']),
               vmax=np.max(c['sta_unscaled']))
    plt.title('Brightest pixel: {}'.format(c['max_i']))
    plt.tight_layout(pad=5, h_pad=1, w_pad=1.8)
    plt.show()
#    plt.savefig(savepath, dpi=200, bbox_inches='tight')
#    plt.close()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 11:22:39 2017

@author: ycan
"""
import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
import warnings

from .. import frametimes as ft
from .. import io as iof
from .. import misc as msc
from .. import randpy
from ..plot import util as plf


def checkerflicker(exp_name, stimulusnr, clusterstoanalyze=None,
                   frametimingsfraction=None, cutoff=4):
    """
    Analyzes checkerflicker data. Saves the results in .npz and .h5
    formats.

    Parameters:
    ----------
        exp_name:
            Experiment name.
        stimulusnr:
            Number of the stimulus to be analyzed.
        clusterstoanalyze:
            Number of clusters should be analyzed. Default is None.

            First N cells will be analyzed if this parameter is given.
            In case of long recordings it might make sense to first
            look at a subset of cells before starting to analyze
            the whole dataset.

        frametimingsfraction:
            Fraction of the recording to analyze. Should be a number
            between 0 and 1. e.g. 0.3 will analyze the first 30% of
            the whole recording.
        cutoff:
           Worst rating that is wanted for the analysis. Default
           is 4. The source of this value is manual rating of each
           cluster.
    """
    exp_dir = iof.exp_dir_fixer(exp_name)

    stimname = iof.getstimname(exp_dir, stimulusnr)

    exp_name = os.path.split(exp_dir)[-1]

    clusters, metadata = iof.read_spikesheet(exp_dir, cutoff=cutoff)

    # Check that the inputs are as expected.
    if clusterstoanalyze:
        if clusterstoanalyze > len(clusters[:, 0]):
            warnings.warn('clusterstoanalyze is larger '
                          'than number of clusters in dataset. '
                          'All cells will be included.')
            clusterstoanalyze = None
    if frametimingsfraction:
        if not 0 < frametimingsfraction < 1:
            raise ValueError('Invalid input for frametimingsfraction: {}. '
                             'It should be a number between 0 and 1'
                             ''.format(frametimingsfraction))

    scr_width = metadata['screen_width']
    scr_height = metadata['screen_height']

    refresh_rate = metadata['refresh_rate']

    parameters = iof.read_parameters(exp_dir, stimulusnr)

    stx_h = parameters['stixelheight']
    stx_w = parameters['stixelwidth']

    # Check whether any parameters are given for margins, calculate
    # screen dimensions.
    marginkeys = ['tmargin', 'bmargin', 'rmargin', 'lmargin']
    margins = []
    for key in marginkeys:
        margins.append(parameters.get(key, 0))

    # Subtract bottom and top from vertical dimension; left and right
    # from horizontal dimension
    scr_width = scr_width-sum(margins[2:])
    scr_height = scr_height-sum(margins[:2])

    nblinks = parameters['Nblinks']
    bw = parameters.dict('blackwhite', False)

    # Gaussian stimuli are not supported yet, we need to ensure we
    # have a black and white stimulus
    if bw is not True:
        raise ValueError('Gaussian stimuli are not supported yet!')

    seed = parameters.get('seed', -10000)

    sx, sy = scr_height/stx_h, scr_width/stx_w

    # Make sure that the number of stimulus pixels are integers
    # Rounding down is also possible but might require
    # other considerations.
    if sx % 1 == 0 and sy % 1 == 0:
        sx, sy = int(sx), int(sy)
    else:
        raise ValueError('sx and sy must be integers')

    filter_length, frametimings = ft.ft_nblinks(exp_dir, stimulusnr, nblinks,
                                                refresh_rate)

    savefname = str(stimulusnr)+'_data'

    if clusterstoanalyze:
        clusters = clusters[:clusterstoanalyze, :]
        print('Analyzing first %s cells' % clusterstoanalyze)
        savefname += '_'+str(clusterstoanalyze)+'cells'
    if frametimingsfraction:
        frametimingsindex = int(len(frametimings)*frametimingsfraction)
        frametimings = frametimings[:frametimingsindex]
        print('Analyzing first {}% of'
              ' the recording'.format(frametimingsfraction*100))
        savefname += '_'+str(frametimingsfraction).replace('.', '')+'fraction'
    frame_duration = np.average(np.ediff1d(frametimings))
    total_frames = frametimings.shape[0]

    all_spiketimes = []
    # Store spike triggered averages in a list containing correct shaped
    # arrays
    stas = []

    for i in range(len(clusters[:, 0])):
        spiketimes = iof.read_raster(exp_dir, stimulusnr,
                                     clusters[i, 0], clusters[i, 1])

        spikes = msc.binspikes(spiketimes, frametimings)
        all_spiketimes.append(spikes)
        stas.append(np.zeros((sx, sy, filter_length)))

    # Empirically determined to be best for 32GB RAM
    desired_chunk_size = 21600000

    # Length of the chunks (specified in number of frames)
    chunklength = int(desired_chunk_size/(sx*sy))

    chunksize = chunklength*sx*sy
    nrofchunks = int(np.ceil(total_frames/chunklength))

    print(f'\nAnalyzing {stimname}.\nTotal chunks: {nrofchunks}')

    time = startime = datetime.datetime.now()

    quals = np.zeros(len(stas))

    for i in range(nrofchunks):
        randnrs, seed = randpy.ranb(seed, chunksize)
        # Reshape and change 0's to -1's
        stimulus = np.reshape(randnrs, (sx, sy, chunklength), order='F')*2-1
        del randnrs
        # Range of indices we are interested in for the current chunk
        if (i+1)*chunklength < total_frames:
            chunkind = slice(i*chunklength, (i+1)*chunklength)
            chunkend = chunklength
        else:
            chunkind = slice(i*chunklength, None)
            chunkend = total_frames - i*chunklength

        for k in range(filter_length, chunkend-filter_length+1):
            stim_small = stimulus[:, :, k-filter_length+1:k+1][:, :, ::-1]
            for j in range(clusters.shape[0]):
                spikes = all_spiketimes[j][chunkind]
                if spikes[k] != 0:
                    stas[j] += spikes[k]*stim_small
        qual = np.array([])
        for c in range(clusters.shape[0]):
            qual = np.append(qual, msc.staquality(stas[c]))
        quals = np.vstack((quals, qual))

        if i == 1:
            print('Estimated analysis time: '
                  f'{msc.timediff(time)*(nrofchunks)}\n')
        time = datetime.datetime.now()

    # Remove the first row which is full of random nrs.
    quals = quals[1:, :]

    max_inds = []
    spikenrs = np.array([spikearr.sum() for spikearr in all_spiketimes])

    for i in range(clusters.shape[0]):
        stas[i] = stas[i]/spikenrs[i]
        # Find the pixel with largest absolute value
        max_i = np.squeeze(np.where(np.abs(stas[i])
                                    == np.max(np.abs(stas[i]))))
        # If there are multiple pixels with largest value,
        # take the first one.
        if max_i.shape != (3,):
            try:
                max_i = max_i[:, 0]
            # If max_i cannot be found just set it to zeros.
            except IndexError:
                max_i = np.array([0, 0, 0])

        max_inds.append(max_i)

    print(f'Completed. Total elapsed time: {msc.timediff(startime)}\n' +
          f'Finished on {datetime.datetime.now().strftime("%A %X")}')

    savepath = os.path.join(exp_dir, 'data_analysis', stimname)
    if not os.path.isdir(savepath):
        os.makedirs(savepath, exist_ok=True)
    savepath = os.path.join(savepath, savefname)

    keystosave = ['clusters', 'frametimings', 'all_spiketimes',
                  'frame_duration', 'max_inds', 'nblinks', 'stas',
                  'stx_h', 'stx_w', 'total_frames', 'sx', 'sy',
                  'filter_length', 'stimname', 'exp_name', 'spikenrs',
                  'clusterstoanalyze', 'frametimingsfraction', 'cutoff',
                  'quals', 'nrofchunks', 'chunklength']
    datadict = {}

    for key in keystosave:
        datadict[key] = locals()[key]

    np.savez(savepath, **datadict)

    t = (np.arange(nrofchunks)*chunklength*frame_duration)/refresh_rate
    qmax = np.max(quals, axis=0)
    qualsn = quals/qmax[np.newaxis, :]

    ax = plt.subplot(111)
    ax.plot(t, qualsn, alpha=0.3)
    plt.ylabel('Z-score of center pixel (normalized)')
    plt.xlabel('Minutes of stimulus analyzed')
    plt.ylim([0, 1])
    plf.spineless(ax, 'tr')
    plt.title(f'Recording duration optimization\n{exp_name}\n {savefname}')
    plt.savefig(savepath+'.svg', format='svg')
    plt.close()
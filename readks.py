#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""

import os

import numpy as np
import pandas as pd
import h5py

import analysis_scripts as asc

#%%
def read_spiketimes(folder):
    spiketimes_insamples = np.load(os.path.join(folder,
                                                'spike_times.npy')).squeeze()
    return spiketimes_insamples


def read_spikeclusters(folder):
    return np.load(os.path.join(folder, 'spike_clusters.npy'))


def read_sample_perstim(folder):
    with h5py.File(os.path.join(folder, 'bininfo.mat'), mode='r') as f:
        stimsamples = f['bininfo']['stimsamples'][()].astype(int)
    return stimsamples

#%%
def get_stim_boundaries(folder):
    return np.cumsum(read_sample_perstim(folder))


def cluster_spikes_perstim(folder, stimnr, spiketimes, spikeclusters):
    read_bounds = get_stim_boundaries(folder)
    bounds = np.zeros(read_bounds.shape[0]+1)
    bounds[1:] = read_bounds
    stimrange = np.logical_and(spiketimes > bounds[stimnr-1],
                               spiketimes < bounds[stimnr])
    return spiketimes[stimrange], spikeclusters[stimrange]

def read_params(folder):
    with open(os.path.join(folder, 'params.py'), 'r') as f:
        lines = f.readlines()
    return lines

def read_samplingrate(folder):
    lines = read_params(folder)
    for line in lines:
        if line.startswith('sample_rate'):
            return int(line.split('= ')[1].split('.')[0])


def get_goodcells(folder):
    df = read_infofile(folder)
    df = add_clusternumbers(df)
    return df


def goodcell_ids(folder):
    return get_goodcells(folder).id


def goodspikes_boolmask(folder):
    goodcells = get_goodcells(folder)
    spc = read_spikeclusters(folder)
    goodspikes = np.in1d(spc, goodcells)
    return goodspikes


def onlygoodclustersandspikes(folder):
    spt = read_spiketimes(folder)
    spc = read_spikeclusters(folder)
    goodspikes = goodspikes_boolmask(folder)
    return spt[goodspikes], spc[goodspikes]


def chopintostimuli(folder):
    spiketimes, spikeclusters = onlygoodclustersandspikes(folder)
    stimbounds = get_stim_boundaries(folder)
    nstim = stimbounds.shape[0]
    spikes = []
    for i in range(nstim):
        spt_stim, spc_stim = cluster_spikes_perstim(folder, i+1,
                                                    spiketimes,
                                                    spikeclusters)
        if i == 0:
            begin_sample = 0
        else:
            begin_sample = stimbounds[i-1]

        # Start each stimulus from zero
        spt_stim = spt_stim - begin_sample
        spikes.append([spt_stim, spc_stim])
    return spikes


def assign_spikes_toclusters(folder, spikes):
    spt_stim, spc_stim = spikes
    goodcells = np.array(get_goodcells(folder), dtype=int)
    ncells = goodcells.shape[0]
    all_cl = [[] for i in range(ncells)]
    for i in range(ncells):
        cellspikes = spt_stim[spc_stim == goodcells[i]]
        all_cl[i].extend(cellspikes)
    return all_cl


def save_spikes_perstimuli(folder, spikes, goodcells):
    """
    spikes should be list of lists, containing [spiketimes, spikeclusters] for each stimuli
    goodcells should be a dataframe with only the good cells
    """
    os.makedirs(os.path.join(folder, 'spikes'), exist_ok=True)
    for i, stimspikes in enumerate(spikes):
        spikespercell_stim = []
        for aydi in goodcells.id:
            # Find all spikes belonging to one cluster in a stimulus
            single_cell_spikes = stimspikes[0][stimspikes[1] == aydi]
            # Convert from samples to seconds
            spikespercell_stim.append(single_cell_spikes / read_samplingrate(folder))

        np.savez(os.path.join(folder, 'spikes', f'{i+1}.npz'),
                 spikes=spikespercell_stim)


def generate_clusternumbers(channels):
    """
    Generate consecutive cluster numbers for each channel, similar
    to IGOR cluster naming.
    """
    ch_prev = None
    cl_prev = 0
    clusters = []
    for ch in channels:
        if ch == ch_prev:
            cl_prev += 1
            clusters.append(cl_prev)
        else:
            cl_prev = 1
            clusters.append(cl_prev)
        ch_prev = ch
    return clusters


def pick_goodcells(dataframe):
    return dataframe[dataframe.group == 'good']


def pick_usefulcols(dataframe):
    return dataframe.loc[:, ['id', 'ch', 'quality', 'comment', 'group', 'n_spikes']]


def sortbychannel(dataframe):
    """
    Sort by channel, then id to make ordering as deterministic as possible.
    """
    return dataframe.sort_values(['ch', 'id'])


def add_clusternumbers(dataframe):
    df = sortbychannel(pick_usefulcols(pick_goodcells(dataframe)))
    clusters = generate_clusternumbers(df.ch)
    # Insert the cluster column next to channel column
    col_ind_cha = int(np.where(df.columns == 'ch')[0])
    df.insert(col_ind_cha+1, 'cluster', clusters)
    return df


def read_infofile(folder):
    """
    Returns the info file as written by phy
    """
    infofile = pd.read_csv(os.path.join(folder, 'cluster_info.tsv'),
                       sep='\t', header=0)
    # Change from index zero to index one
    infofile.loc[:, 'ch'] +=1
    return infofile


def preprocess_ks(folder):
    """
    Main function to extract the spikes per cluster and parse the
    information file.
    """
    goodcells = add_clusternumbers(read_infofile(folder))

    spikes = chopintostimuli(folder)

    save_spikes_perstimuli(folder, spikes, goodcells)
    goodcells.to_csv(os.path.join(folder, 'spikes', 'clusters.csv'))


def read_clusters(folder):
    """
    Returns the cleaned up clusters file after excluding noise cells
    and dropping some columns
    """
    # pd.read_csv(os.path.join(folder, 'cluster_info.tsv'),
    #                    sep='\t', header=0)

    infofile = pd.read_csv(os.path.join(folder, 'spikes', 'clusters.csv'))
    infofile.loc[:, 'ch'] +=1
    return infofile

def clusters_spikesheet(folder):
    """
    Generate the clusters array to emulate asc.read_spikesheet behavior
    """
    clusters_pd = read_clusters(asc.kilosorted_path(folder))
    return np.array(clusters_pd.loc[:, ['ch', 'cluster', 'quality']], dtype=int)


def clusters_metadata(folder):
    """
    Generate metadata dictionary to emulate asc.read_spikesheet behavior
    """

    def convert(val):
        """
        Cast from string to appropriate type
        """
        constructors = [int, float, str, bool]
        for c in constructors:
            try:
                return c(val)
            except ValueError:
                pass

    try:
        metadata = asc.read_spikesheet(asc.kilosorted_path(folder), onlymetadata=True)
        return metadata
    except FileNotFoundError:
        pass

    params = read_params(asc.kilosorted_path(folder))

    params_dict = {}

    for el in params:
        keyval = el.split('=')
        key, val = keyval[0].strip(), convert(keyval[1].strip())
        if key == 'sample_rate':
            # Naming is expected to be specifically sample_freq
            key = 'sample_freq'
            val = int(val)
        elif key == 'hp_filtered':
            val = val == 'True'
        params_dict.update({key: val})

    return params_dict


def read_spikesheet_ks(folder):
    return clusters_spikesheet(folder), clusters_metadata(folder)


def load_spikes(folder, stimnr):
    """
    Loads the spike times in seconds for all cells in a single stimulus.

    An object array containing a numpy array for each cell is returned.
    """
    data = np.load(os.path.join(folder, 'spikes', f'{stimnr}.npz'),
                   allow_pickle=True)['spikes']
    return data


def check_clusters(folder):
    """
    Check that all clusters are labeled as expected.

    There should be no noise clusters with a quality rating.
    There should be no good clusters without a quality rating.
    """
    df = read_infofile(folder)

    noise_w_qual = pick_usefulcols(df[(df.group == 'noise') & ~df.quality.isna()])

    good_wo_qual = pick_usefulcols(df[(df.group == 'good') & df.quality.isna()])

    if noise_w_qual.size != 0:
        print('Warning: following clusters are labeled as noise but have quality ratings:')
        print(noise_w_qual)
    if good_wo_qual.size != 0:
        print('Warning: following clusters are labeled as good but have no quality rating:')
        print(good_wo_qual)

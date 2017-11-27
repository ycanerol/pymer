#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 15:54:03 2017

@author: ycan

Collection of analysis functions
"""
import numpy as np


def read_ods(ods_fpath, cutoff=4):
    """
    Read metadata and cluster information from .ods file (manually
    created during spike sorting), return good clusters.

    Parameters:
    -----------
        ods_fpath:
            Full path of the .ods file to be used

        cutoff:
            Worst rating that is wanted for the analysis. Default
            is 4. The source of this value is manual rating of each
            cluster.

    Returns:
    --------
        clusters:
            Channel number, cluster number and rating of those
            clusters that match the cutoff criteria in a numpy array.

        metadata:
            Information about the experiment in a dictionary.

    Notes:
    ------
    The script assumes adherence to the defined cell location for
    metadata and cluster information. If changed weird errors might
    come up.

    An accompanying read-only template .ods file is also created.

    Empty cells in the .ods file are likely to cause problems.

    First version: 2017-11-21 by Yunus
    """
    import pyexcel_ods as pyxo
    clusters = pyxo.get_data(ods_fpath,
                             start_row=5, row_limit=400,
                             start_column=0, column_limit=6)
    metadata = pyxo.get_data(ods_fpath,
                             start_row=0, row_limit=2,
                             start_column=0, column_limit=25)
    clusters = np.array(clusters['Sheet1'])
    # Get rid of unneeded columns using numpy advanced indexing
    clusters = clusters[:, [0, 4, 5]]
    # Fill the gaps and convert to int
    for i in range(len(clusters[:, 0])):
        if clusters[i, 0] != '':
            nr = clusters[i, 0]
        else:
            clusters[i, 0] = nr
    clusters = clusters.astype(int)

    # Filter according to quality cutoff
    clusters = clusters[clusters[:, 2] <= cutoff]

    metadata = np.array(metadata['Sheet1'])
    if len(metadata[0]) == len(metadata[1])+1:
        # Handle edge case if the last cell of second row is empty.
        metadata[1].append('')

    metadata_dict = {}
    for i in range(len(metadata[0])):
        metadata_dict[metadata[0][i]] = metadata[1][i]

    return clusters, metadata_dict

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 15:54:03 2017

@author: ycan

Collection of analysis functions
"""
import numpy as np


def read_ods(experiment_dir, cutoff=4, defaultpath=True):
    """
    Read metadata and cluster information from .ods file (manually
    created during spike sorting), return good clusters.

    Parameters:
    -----------
        experiment_dir:
            Experiment directory that contains the .ods file. The function
            looks for /spike_sorting.ods under this directory.
        cutoff:
            Worst rating that is wanted for the analysis. Default
            is 4. The source of this value is manual rating of each
            cluster.
        defaultpath:
            Whether to add '/spike_sorting.ods' to find the file. If False,
            the full path to .ods should be supplied in experiment_dir.

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

    if defaultpath:
        filepath = experiment_dir + '/spike_sorting.ods'
    else:
        filepath = experiment_dir

    import pyexcel_ods as pyxo
    clusters = pyxo.get_data(filepath,
                             start_row=4, row_limit=400,
                             start_column=0, column_limit=6)
    metadata = pyxo.get_data(filepath,
                             start_row=0, row_limit=2,
                             start_column=0, column_limit=25)
    clusters = np.array(clusters['Sheet1'])
    # Get rid of unneeded columns using numpy advanced indexing
    try:
        clusters = clusters[:, [0, 4, 5]]
    except IndexError:
        # This happens when np.array() cannot convert clusters into an np.array
        # containing strings; instead it returns an object array which cannot
        # be indexed this way.
        # If you're getting this error, check that there wasn't any ratings
        # columns that are left blank.
        raise ValueError('.ods file is missing information! Check that '
                         'all clusters have a rating!')
        return None, None
    # The channels with multiple clusters have an empty line after the first
    # line. Fill the empty lines using the first line of each channel.
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


def getframetimes(experiment_dir, stimnr, defaultpath=True, threshold=75,
                  sampling_rate=10000, plotting=False,
                  time_offset=25, zeroADvalue=32768,
                  microvoltsperADunit=25625/2048):
    """
    Extract frame timings from the triggered signal recorded alongside the
    MEA data.

    Typically the timing data is in the file <stimulusnr>_253.bin. It comes in
    pulses; onset and offset of the pulse denotes different things depending
    on the stimulus code.

    The most typical case is that pulse onset corresponds to when a frame comes
    on screen, and the pulse is turned off at the next frame (even if the
    consecutive frame is identical). In this case the duration of the pulse
    will be 1000 ms/refresh rate; which is ~16.6 ms for 60 Hz refresh rate.
    For these type of stimuli, using only pulse onsets is sufficient.

    For some types of stimuli, the pulse offset is also important, for
    these cases pulse onsets and offsets need to be used together.

    There is also a delay between the pulse and the frame actually being
    displayed, which should be accounted for. The delay is 25 ms for setup
    Bilbo.

    Parameters:
    ----------
        experiment_dir:
            Path to the binary file. '.../<stimulus_nr>_253.bin'
        stimnr:
            Number of the stimulus, to find /<stimulus_nr>_253.bin for the
            stimulus of interest.
        defaultpath:
            Whether to use experiment_dir+'RawChannels/'+stim_nr+'_253.bin'
            to access the frametimings binary file. Default is True. If False
            full path should be passed with experiment_dir; stim_nr is not
            important in this case, just pass empty.
        threshold:
            The threshold in milivolts for the trigger signal. Default is
            75 mV.
        sampling_rate:
            Sampling rate of the recording in Hz.
        plotting:
            Whether to plot the whole trace and signal on-offsets. Slow for
            long recordings and frequent pulses (e.g. checkerflicker). Default
            is False.
        time_offset:
            The delay between the pulse generation and the disply actually
            updating. Default is 25 ms.
        zeroADvalue:
            The zero point of the analog digital conversion. Copied directly
            from frametimings10.m by Norma(?). Default is 32768.
        microvoltsperADunit:
            Conversion factor between the readout and voltage. Resulting
            conversion result is in milivolts. Default is 25625/2048.


    Returns:
    -------
        frametimings_on:
            List of times in miliseconds where a pulse started, corresponding
            to a frame update. Corrected for the monitor delay by time_offset.
        frametimings_off:
            List of times in miliseconds where a pulse ended. Not to be used
            frequently, only if a particular stimulus requires it.


    """
    import struct
    import os

    if defaultpath:
        filepath = os.path.join(experiment_dir, 'RawChannels',
                                str(stimnr)+'_253.bin')
    else:
        filepath = experiment_dir

    with open(filepath, mode='rb') as file:  # b is important -> binary
        fileContent = file.read()
    # %%

    # The header contains the length of the recording as a unsigned 32 bit int
    length = struct.unpack('I', fileContent[:4])[0]
    # The rest of the binary data is the voltage trace, as unsigned 16 bit int
    voltage_raw = np.array(struct.unpack('H'*length, fileContent[16:]))

    # Convert units to microvolts, using constants from Norma(?)'s script
    voltage = (voltage_raw - zeroADvalue) / microvoltsperADunit
    # Set the baseline value to zero
    voltage = voltage - voltage[voltage < threshold].mean()
    # %%
    time = np.arange(length) / (sampling_rate * 1e-3)  # In miliseconds
    time = time + time_offset  # Correct for the time delay

    def thr_cros(array, threshold):
        size = array.shape[0]
        oncross = np.array([False]*size)
        offcross = np.array([False]*size)
        for i in range(size-1):
            if array[i-1] < threshold and array[i] > threshold:
                oncross[i] = True
            if array[i-1] > threshold and array[i] < threshold:
                offcross[i] = True
        return oncross, offcross

    print('Total recording time: {:6.1f} seconds'
          ' (= {:3.1f} minutes)'.format(length/sampling_rate,
                                        (length/sampling_rate)/60))

    onsets, offsets = thr_cros(voltage, threshold)
    if onsets.sum() != offsets.sum():
        print('Number of pulse onset and offsets are not equal!')
    if plotting:
        import matplotlib.pyplot as plt
        # Plot the whole voltage trace
        plt.figure(figsize=(10, 10))
        plt.plot(time, voltage)
        plt.show()

        # Put all stimulus onset and offsets on top of each other
        # This part takes very long time for long recordings
        plt.figure()
        for i in range(onsets.shape[0]):
            if onsets[i]:
                plt.subplot(211)
                plt.plot(voltage[i-2:i+3])
            if offsets[i]:
                plt.subplot(212)
                plt.plot(voltage[i-2:i+3])
        plt.show()
        plt.close()

    frametimings_on = time[onsets]
    frametimings_off = time[offsets]

    return frametimings_on, frametimings_off


def read_raster(experiment_dir, stimnr, channel, cluster, defaultpath=True,
                alternatepath=''):
    """
    Return the spike times from the specified raster file.

    Use defaultpath=False if the raster directory is not
    experiment_dir + '/results/rasters/'. In this case alternatepath will be
    appended to experiment_dir.
    """
    import os

    if defaultpath:
        r = os.path.join(experiment_dir, 'results/rasters/')
    else:
        r = os.path.join(experiment_dir, alternatepath)
    s = str(stimnr)
    c = str(channel)
    fullpath = r + s + '_SP_C' + c + '{:0>2}'.format(cluster) + '.txt'
    spike_file = open(fullpath)
    spike_times = np.array([float(line) for line in spike_file])
    spike_file.close()

    return spike_times

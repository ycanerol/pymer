#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 15:54:03 2017

@author: ycan

Collection of analysis functions
"""
import numpy as np
import os
import glob
import iofuncs as iof


def read_ods(exp_name, cutoff=4, defaultpath=True):
    """
    Read metadata and cluster information from .ods file (manually
    created during spike sorting), return good clusters.

    Parameters:
    -----------
        exp_name:
            Experiment name for the directory that contains the
            .ods file. The function looks for /spike_sorting.ods
            under this directory.
        cutoff:
            Worst rating that is wanted for the analysis. Default
            is 4. The source of this value is manual rating of each
            cluster.
        defaultpath:
            Whether to add '/spike_sorting.ods' to find the file. If False,
            the full path to .ods should be supplied in exp_dir.

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
    exp_dir = iof.exp_dir_fixer(exp_name)
    if defaultpath:
        filepath = exp_dir + '/spike_sorting.ods'
    else:
        filepath = exp_dir

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


def readframetimes(exp_name, stimnr, returnoffsets=False):
    """
    Reads the extracted frame times from exp_dir/frametimes folder.

    Parameters:
    ----------
        exp_name:
            Experiment name to be used.
        stimnr:
            Order of the stimulus of interest.
        returnoffsets:
            Whether to return the offset times as well as onset times. If True,
            two arrays are returned.

    Returns:
    -------
        frametimings_on:
            List of times in seconds where a pulse started, corresponding
            to a frame update. Corrected for the monitor delay by time_offset.
        frametimings_off:
            List of times in seconds where a pulse ended. Only returned if
            returnoffsets is True. Not to be used frequently, only if a
            particular stimulus requires it.
    """
    exp_dir = iof.exp_dir_fixer(exp_name)

    filepath = os.path.join(exp_dir, 'frametimes', str(stimnr)+'_*.npz')
    filename = glob.glob(filepath)[0]
    f = np.load(filename)

    frametimings_on = f['f_on']

    if returnoffsets:
        frametimings_off = f['f_off']
        return frametimings_on, frametimings_off
    else:
        return frametimings_on


def saveframetimes(exp_name, **kwargs):
    """
    Save all frametiming data for one experiment.

    Parameters:
    ----------
        exp_name:
            Experiment name.
    """
    exp_dir = iof.exp_dir_fixer(exp_name)

    for i in range(1, 100):
        try:
            stimname = np.sort(glob.glob('{}_*.mcd'.format(i)))[0]
            stimname = stimname.strip('.mcd')
            print(stimname)
        except IndexError:
            break
        f_on, f_off = extractframetimes(exp_dir, i, **kwargs)

        savepath = os.path.join(exp_dir, 'frametimes')

        if not os.path.exists(savepath):
            os.mkdir(savepath)

        np.savez(os.path.join(savepath, stimname+'_frametimes'),
                 f_on=f_on,
                 f_off=f_off)


def extractframetimes(exp_name, stimnr, threshold=75,
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
        exp_dir:
            Experiment name.
        stimnr:
            Number of the stimulus, to find /<stimulus_nr>_253.bin for the
            stimulus of interest.
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
            List of times in seconds where a pulse started, corresponding
            to a frame update. Corrected for the monitor delay by time_offset.
        frametimings_off:
            List of times in seconds where a pulse ended. Only returned if
            returnoffsets is True. Not to be used frequently, only if a
            particular stimulus requires it.

    """
    import struct

    exp_dir = iof.exp_dir_fixer(exp_name)
    filepath = os.path.join(exp_dir, 'RawChannels',
                            str(stimnr)+'_253.bin')

    with open(filepath, mode='rb') as file:  # b is important -> binary
        fileContent = file.read()

    # The header contains the length of the recording as a unsigned 32 bit int
    length = struct.unpack('I', fileContent[:4])[0]
    # The rest of the binary data is the voltage trace, as unsigned 16 bit int
    voltage_raw = np.array(struct.unpack('H'*length, fileContent[16:]))

    # Convert units to microvolts, using constants from Norma(?)'s script
    voltage = (voltage_raw - zeroADvalue) / microvoltsperADunit
    # Set the baseline value to zero
    voltage = voltage - voltage[voltage < threshold].mean()

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

    # Get the times where on-offsets happen and convert from miliseconds
    # to seconds
    frametimings_on = time[onsets]/1000
    frametimings_off = time[offsets]/1000

    return frametimings_on, frametimings_off


def read_raster(exp_name, stimnr, channel, cluster, defaultpath=True):
    """
    Return the spike times from the specified raster file.

    Use defaultpath=False if the raster directory is not
    exp_dir + '/results/rasters/'. In this case pass the full
    path to the raster with exp_dir.
    """
    exp_dir = iof.exp_dir_fixer(exp_name)
    if defaultpath:
        r = os.path.join(exp_dir, 'results/rasters/')
    else:
        r = exp_dir
    s = str(stimnr)
    c = str(channel)
    fullpath = r + s + '_SP_C' + c + '{:0>2}'.format(cluster) + '.txt'
    spike_file = open(fullpath)
    spike_times = np.array([float(line) for line in spike_file])
    spike_file.close()

    return spike_times


def read_parameters(exp_name, stimulusnr, defaultpath=True):
    """
    Reads the parameters from stimulus files

    Parameters:
    -----------
    exp_name:
        Experiment name. The function will look for 'stimuli'
        folder under the experiment directory.
    stimulusnr:
        The order of the stimulus. The function will open the files with the
        file name '<stimulusnr>_*' under the stimulus directory.
    defaultpath:
         Whether to use exp_dir+'/stimuli/' to access the stimuli
         parameters. Default is True. If False full path to stimulus folder
         should be passed with exp_dir.

    Returns:
    -------
    parameters:
        Dictionary containing all of the parameters. Parameters are
        are variable for different stimuli; but for each type, at least file
        name and stimulus type are returned.

    For spontaneous activity recordings, an empty text file is expected in the
    stimuli folder. In this case the stimulus type is returned as spontaneous
    activity.

    """
    exp_dir = iof.exp_dir_fixer(exp_name)

    if defaultpath:
        stimdir = os.path.join(exp_dir, 'stimuli')
    else:
        stimdir = exp_dir

    paramfile = glob.glob(os.path.join(stimdir, '{}_*'.format(stimulusnr)))
    if len(paramfile) == 1:
        paramfile = paramfile[0]
    elif len(paramfile) == 0:
        raise IOError('No parameter file that starts with {} exists under'
                      ' the directory: {}'.format(stimulusnr, stimdir))
    else:
        print(paramfile)

        raise ValueError('Multiple files were found starting'
                         ' with {}'.format(stimulusnr))

    f = open(paramfile)
    lines = [line.strip('\n') for line in f]
    f.close()

    parameters = {}

    parameters['filename'] = os.path.split(paramfile)[-1]
    if len(lines) == 0:
        parameters['stimulus_type'] = 'spontaneous_activity'

    for line in lines:
        if len(line) == 0:
            continue
        try:
            key, value = line.split('=')
            key = key.strip(' ')
            value = value.strip(' ')
            try:
                value = float(value)
                if value % 1 == 0:
                    value = int(value)
            except ValueError:
                if value == ' true' or value == 'true':
                    value = True
                elif value == ' false' or value == 'false':
                    value = False

            parameters[key] = value
        except ValueError:
            parameters['stimulus_type'] = line

    return parameters


def binspikes(spiketimes, frametimings):
    """
    Bins spikes according to frametimings, discards spikes before and after
    stimulus presentation

    Parameters:
    ----------
        spiketimes:
            Array containing times of when spikes happen, as output of
            read_rasters()
        frametimings:
            Array containing frametimings as output of readframetimes()
    Returns:
    -------
        spikes:
            Binned array of spikes according to frametimings.

    Notes:
    -----
        Binning according to frametimings results in high count in first and
        last bins since recording is started earlier than stimulus presentation
        and stopped after presentation ends. Therefore the spikes that happen
        before or after stimulus presentation are discarded in the output.
    """
    spikes = np.bincount(np.digitize(spiketimes, frametimings))
    if spikes.shape == (0,):
        # If there are no spikes for a particular cell,
        # set it to an array of zeros.
        spikes = np.zeros((frametimings.shape[0]+1,))
    spikes[0] = 0
    spikes = spikes[:-1]

    # HINT: This might cause problems! Not sure this is the correct way to
    # fix this problem, might cause misalignment of frametimes and spikes
    #
    # If there hasn't been any spikes at the end of the recording, the length
    # of the spikes array will be shorter than the frametimings and this
    # results in not having data at the end of the recording.
    while spikes.shape[0] < frametimings.shape[0]:
        spikes = np.append(spikes, 0)

    return spikes


def staquality(sta):
    """
    Calculates the z-score of the pixel that is furthest away
    from the zero as a measure of STA quality.
    """

    z = (np.max(np.abs(sta)) - sta.mean()) / sta.std()
    return z.astype('float16')

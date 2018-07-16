#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 15:54:03 2017

@author: ycan

Collection of analysis functions
"""
import os
import glob
import numpy as np
import iofuncs as iof
import warnings

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
    if defaultpath:
        exp_dir = iof.exp_dir_fixer(exp_name)
        filepath = exp_dir + '/spike_sorting.ods'
    else:
        filepath = exp_name
    import pyexcel_ods as pyxo
    clusters = pyxo.get_data(filepath,
                             start_row=4, row_limit=400,
                             start_column=0, column_limit=6)
    metadata = pyxo.get_data(filepath,
                             start_row=0, row_limit=2,
                             start_column=0, column_limit=25)

    clusters = clusters['Sheet1']
    # Check for trailing empty elements
    # Iterate backwards because we are removing stuff
    for i in range(len(clusters)):
        el = clusters[len(clusters)-i-1]
        if len(el)==0:
            clusters.remove([])
    clusters = np.array(clusters)
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
    try:
        filename = glob.glob(filepath)[0]
    except IndexError:
        raise ValueError(f'No frametimes file for {stimnr} in {exp_name}.')
    f = np.load(filename)

    frametimings_on = f['f_on']

    if returnoffsets:
        frametimings_off = f['f_off']
        return frametimings_on, frametimings_off
    else:
        return frametimings_on


def saveframetimes(exp_name, forceextraction=False, **kwargs):
    """
    Save all frametiming data for one experiment.

    Nothing will be saved if frametimings files already exist.
    forceextraction parameter can be used to override this behaviour.

    Parameters:
    ----------
        exp_name:
            Experiment name.
    """
    exp_dir = iof.exp_dir_fixer(exp_name)

    for i in range(1, 100):
        alreadyextracted = True
        # If we have already extracted the frametimes, no need to do it twice.
        try:
            readframetimes(exp_dir, i)
        except ValueError as e:
            if str(e).startswith('No frametimes file'):
                alreadyextracted = False
        if forceextraction:
            alreadyextracted = False
        if not alreadyextracted:
            try:
                stimname = iof.getstimname(exp_name, i)
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
                      plotting=False, zeroADvalue=32768):
    """
    Extract frame timings from the triggered signal recorded alongside the
    MEA data.

    Typically the timing data is in the file /<stimulus_nr>_<253 or 61>.bin.
    It comes in pulses; onset and offset of the pulse denotes different
    things depending on the stimulus code.

    The most typical case is that pulse onset corresponds to when a frame comes
    on screen, and the pulse is turned off at the next frame (even if the
    consecutive frame is identical). In this case the duration of the pulse
    will be 1000 ms/refresh rate; which is ~16.6 ms for 60 Hz refresh rate.
    For these type of stimuli, using only pulse onsets is sufficient.

    For some types of stimuli (e.g. 1 blinks), the pulse offset is also
    important, for these cases pulse onsets and offsets need to be
    used together.

    There is also a delay between the pulse and the frame actually being
    displayed, which should be accounted for. This is read from the ODS file.

    Parameters:
    ----------
        exp_dir:
            Experiment name.
        stimnr:
            Number of the stimulus, to find the analog channel for pulses
            for the stimulus of interest.
        threshold:
            The threshold in milivolts for the trigger signal. Default is
            75 mV.
        plotting:
            Whether to plot the whole trace and signal on-offsets. Slow for
            long recordings and frequent pulses (e.g. checkerflicker). Default
            is False.
        zeroADvalue:
            The zero point of the analog digital conversion. Copied directly
            from frametimings10.m by Norma(?). Default is 32768.


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

    # Check the type of array used, this will affect the relevant
    # parameters for extraction.
    # microvoltsperADunit was defined empirically from inspecting the
    # pulse traces from different setups.
    _, metadata = read_ods(exp_dir)
    if metadata['MEA'] == 252:
        binfname = '_253.bin'
        microvoltsperADunit = 2066/244
    elif metadata['MEA'] == 60:
        binfname = '_61.bin'
        microvoltsperADunit = 30984/386
    else:
        raise ValueError('Unknown MEA type.')

    monitor_delay = metadata['monitor_delay(s)']

    sampling_rate = metadata['sampling_freq']

    if sampling_rate not in [10000, 25000]:
        # Sanity check, sampling frequency could be mistyped.
        raise ValueError('Sampling frequency of the recording is not '
                         'in the ODS file is not one of the expected values! '
                         'Check for missing zeros in sampling_freq.')

    filepath = os.path.join(exp_dir, 'RawChannels',
                            str(stimnr)+binfname)

    with open(filepath, mode='rb') as file:  # b is important -> binary
        file_content = file.read()

    # The header contains the length of the recording as a unsigned 32 bit int
    length = struct.unpack('I', file_content[:4])[0]
    # The rest of the binary data is the voltage trace, as unsigned 16 bit int
    voltage_raw = np.array(struct.unpack('H'*length, file_content[16:]))

    # Convert units to microvolts, using constants from Norma(?)'s script
    voltage = (voltage_raw - zeroADvalue) / microvoltsperADunit
    # Set the baseline value to zero
    voltage = voltage - voltage[voltage < threshold].mean()

    time = np.arange(length) / (sampling_rate * 1e-3)  # In miliseconds
    time = time + monitor_delay # Correct for the time delay

    def thr_cros(array, threshold):
        size = array.shape[0]
        oncross = np.array([False]*size)
        offcross = np.array([False]*size)
        for i in range(size-1):
            if array[i-1] < threshold and array[i] > threshold:
                oncross[i] = True
            if array[i-1] > threshold and array[i] < threshold:
                offcross[i] = True
        # If a pulse is interrupted at the end, it will cause a bug
        # that the first element in offcross is True (due to negative
        # index referring to the end.)
        # This fixes that issue.
        offcross[0] = False
        return oncross, offcross

    print('Total recording time: {:6.1f} seconds'
          ' (= {:3.1f} minutes)'.format(length/sampling_rate,
                                        (length/sampling_rate)/60))

    onsets, offsets = thr_cros(voltage, threshold)
    if onsets.sum() != offsets.sum():
        print('Number of pulse onset and offsets are not equal!'
              'The last pulse probably was interrupted. Last pulse'
              ' onset was omitted to fix.')
        onsets[np.where(onsets)[0][-1]] = False
    if plotting:
        import matplotlib.pyplot as plt
        # Plot the whole voltage trace
        plt.figure(figsize=(10, 10))
        plt.plot(time, voltage)
        plt.plot(time[onsets], voltage[onsets], 'gx')
        plt.plot(time[offsets], voltage[offsets], 'rx')

        # Put all stimulus onset and offsets on top of each other
        # This part takes very long time for long recordings
        plt.figure(figsize=(9, 6))
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
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        z = (np.max(np.abs(sta)) - sta.mean()) / sta.std()
    return z.astype('float16')


def stimulisorter(exp_name):
    """
    Read parameters.txt file and return the stimuli type and
    stimuli numbers in a dictionary.
    """
    possible_stim_names = ['spontaneous', 'onoffsteps', 'fff', 'stripeflicker',
                           'checkerflicker', 'directiongratingsequence',
                           'rotatingstripes']
    sorted_stimuli = {key:[] for key in possible_stim_names}
    exp_dir = iof.exp_dir_fixer(exp_name)

    file = open(os.path.join(exp_dir, 'parameters.txt'), 'r')

    for line in file:
        for stimname in possible_stim_names:
            if line.find(stimname) > 0:
                stimnr = int(line.split('_')[0])
                toadd = sorted_stimuli[stimname]
                toadd = toadd.append(stimnr)
    return sorted_stimuli

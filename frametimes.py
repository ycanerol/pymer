#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Frame time operations.
"""
import glob
import numpy as np
import os.path
import scipy.io

from . import io as iof


def read(exp_name, stimnr, returnoffsets=False):
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


def save(exp_name, forceextraction=False, start=None, end=None, **kwargs):
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
    if start is None:
        start = 1
    if end is None:
        end = 100

    for i in range(start, end):
        alreadyextracted = True
        # If we have already extracted the frametimes, no need to do it twice.
        try:
            read(exp_dir, i)
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
            f_on, f_off = extract(exp_dir, i, **kwargs)

            savepath = os.path.join(exp_dir, 'frametimes')

            if not os.path.exists(savepath):
                os.mkdir(savepath)

            np.savez(os.path.join(savepath, stimname+'_frametimes'),
                     f_on=f_on,
                     f_off=f_off)


def extract(exp_name, stimnr, threshold=75, plotting=False, zeroADvalue=32768):
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

    exp_dir = iof.exp_dir_fixer(exp_name)

    # Check the type of array used, this will affect the relevant
    # parameters for extraction.
    # microvoltsperADunit was defined empirically from inspecting the
    # pulse traces from different setups.
    _, metadata = iof.read_spikesheet(exp_dir)
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

    file_content = iof.read_binaryfile(filepath)

    length, voltage_raw = iof.parse_binary(file_content)

    voltage = convert_bin2voltage(voltage_raw, zeroADvalue=zeroADvalue,
                                  microvoltsperADunit=microvoltsperADunit)

    # Set the baseline value to zero
    voltage = voltage - voltage[voltage < threshold].mean()

    time = np.arange(length) / (sampling_rate * 1e-3)  # In miliseconds
    time = time + monitor_delay  # Correct for the time delay

    print('Total recording time: {:6.1f} seconds'
          ' (= {:3.1f} minutes)'.format(length/sampling_rate,
                                        (length/sampling_rate)/60))

    onsets, offsets = detect_threshold_crossing(voltage, threshold)
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


def nblinks(exp_dir, stimulusnr, nblinks, refresh_rate):
    """
    Return the appropriate frametimings array depending on the stimulus
    update frequency.
    Parameters
        nblinks :
            Number of screen frames for each stimulus frame, as defined
            in stimulator program.
        refresh_rate :
            Update frequency of the screen that is used. Typically 60Hz.
    Returns
        filter_length:
            Appropriate length of the temporal filter length for STA
        frametimings :
            Array containing timepoints in seconds where the stimulus
            frame was updated.

    """

    # Both onsets and offsets are required in the case of odd numbered
    # nblinks values.
    if nblinks in [1, 3]:
        ft_on, ft_off = read(exp_dir, stimulusnr, returnoffsets=True)
        # Initialize empty array twice the size of one of them, assign
        # value from on or off to every other element.
        frametimings = np.empty(ft_on.shape[0]*2, dtype=float)
        frametimings[::2] = ft_on
        frametimings[1::2] = ft_off

        if nblinks == 3:
            frametimings = frametimings[::3]

    elif nblinks in [2, 4]:
        frametimings = read(exp_dir, stimulusnr)
        if nblinks == 4:
            # There are two pulses per frame
            frametimings = frametimings[::2]
    else:
        raise ValueError(f'Unexpected value for nblinks: {nblinks}')
    # Set the filter length to ~600 ms, this is typically the longest
    # temporal filter one needs. The exact number is chosen to have a
    # round filter_length for nblinks= 1, 2, 4
    filter_length = np.int(np.round(.666*refresh_rate/nblinks))
    return filter_length, frametimings


def loadfrommat(exp_name):
    """
    Extract frame times from .mat files. Needed for analyzing data
    from other people, binary files containing the frame time pulses
    are not usually available.

    The converted frametime files are corrected for monitor delay.
    """
    exp_dir = iof.exp_dir_fixer(exp_name)

    _, metadata = iof.read_spikesheet(exp_dir)
    monitor_delay = metadata['monitor_delay(s)']

    for i in range(1, 100):
        try:
            name = iof.getstimname(exp_dir, i)
        except IndexError as e:
            if str(e).startswith('Stimulus'):
                break
            else:
                raise

        matfile = os.path.join(exp_dir, 'frametimes',
                               name + '_frametimings.mat')

        try:
            f = scipy.io.matlab.loadmat(matfile)
            ftimes = f['ftimes'][0, :]
        except NotImplementedError:
            import h5py
            with h5py.File(matfile, mode='r') as f:
                ftimes = f['ftimes'][:]
                if len(ftimes.shape) != 1:
                    ftimes = ftimes.flatten()

        ftimes = (ftimes/1000)+monitor_delay

        np.savez(os.path.join(exp_dir, 'frametimes', name + '_frametimes'),
                 f_on=ftimes)
        print(f'Converted and saved frametimes for {name}')


def fromnpztomat(exp_name, savedir=None):
    """
    Convert frametime files in .npz to .mat for interoperability
    with MATLAB users.

    savedir
    """
    exp_dir = iof.exp_dir_fixer(exp_name)

    _, metadata = iof.read_spikesheet(exp_dir)
    monitor_delay = metadata['monitor_delay(s)']

    for i in range(1, 100):
        print(i)
        try:
            ft_on, ft_off = read(exp_name, i, returnoffsets=True)
        except ValueError as e:
            if str(e).startswith('No frametimes'):
                break
            else:
                raise
        # Convert to milliseconds b/c that is the convertion in MATLAB scripts
        ft_on = (ft_on - monitor_delay)*1000
        ft_off = (ft_off - monitor_delay)*1000

        stimname = iof.getstimname(exp_dir, i)

        if savedir is None:
            savedir = os.path.join(exp_dir, 'frametimes')
        savename = os.path.join(savedir, stimname)
        print(savename)
        scipy.io.savemat(savename+'_frametimings',
                         {'ftimes': ft_on,
                          'ftimes_offsets': ft_off},
                         appendmat=True)


def convert_bin2voltage(voltage_raw, zeroADvalue=32768,
                        microvoltsperADunit=2066/244):
    """
    Converts raw voltage data in binary format into units of microvolts.

    Helper function for frametimes.extract but also usable on its own.
    """
    voltage = (voltage_raw - zeroADvalue) / microvoltsperADunit
    return voltage


def detect_threshold_crossing(array, threshold):
    """
    Find threshold crossings in a given array. This is a helper function
    for extract channels.
    """
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

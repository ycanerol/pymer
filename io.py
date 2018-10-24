#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions related to reading/writing files.
"""
import collections
import glob
import json
import numpy as np
import os
from pathlib import Path
import pyexcel
import re
import scipy.io
import struct
import sys


# Some variables (e.g. STAs) are stored as lists originally
# but saving to and loading from HDF/npz file converts them to
# numpy arrays with one additional dimension. To revert
# this, we need to turn them back into lists with list()
# function. Any variables that is normally a list should be
# kept in list_of_lists.

list_of_lists = ['stas', 'max_inds', 'all_frs', 'all_parameters', 'fits']


def exp_dir_fixer(exp_name):
    """
    Convert short experiment name into full path. If input is already
    a full path it will be returned as is.

    Following are some valid inputs:
        <valid_prefix>_20171122_fe_re_fp
        20171122_fe_re_fp
        20171122_f
        20171122
    They all will be turned into the same full path:
        <root_experiment_dir>/<prefix>_20171122_252MEA_fr_re_fp

    """
    if os.path.isdir(exp_name):
        # If the experiment name is already a path nothing needs to be done.
        return exp_name

    if config('root_experiment_dir') is None:
        raise Exception('Root experiment directory is not set. See User '
                        'Configuration section on README for instructions.')

    exp_dir = str(exp_name)
    for s in [''] + config('experiment_prefixes'):
        exp_name = s + exp_name
        if not os.path.isdir(exp_dir):
            exp_dir = os.path.join(config('root_experiment_dir'),
                                   exp_name)
            if not os.path.isdir(exp_dir):

                files = glob.glob(exp_dir+'*')

                if len(files) > 1:
                    raise ValueError('Multiple folders'
                                     'found matching'
                                     ' pattern: {}\n {}'.format(exp_name,
                                                                files))
                elif len(files) == 0:
                    if exp_name[0] == '2':
                        continue
                    raise ValueError('No folders matching'
                                     ' pattern: {}'.format(exp_dir))
                else:
                    exp_dir = files[0]
    return exp_dir


def loadh5(path):
    """
    Load data from h5 file in a dictionary.

    Usage:
        data = loadh5(('20171116', 6))
        stas = data['stas']

    Usage:
        locals().update(loadh5(path))

    h5py module returns an HDF file object in memory when data is
    read. This is not convenient for data manipulation, variables
    should be in the active namespace. For this, loadh5 function
    reads the HDF file and returns a dictionary containing all
    the variable and variable name pairs. This dictionary then
    can be used for loading the variables into workspace.

    Parameters
        path:
            Full path to the .h5 file to be read.

            It can also be a tuple containing experiment name and stimulus
            number, in which case _data.h5 file will be used.
    Returns
        data_in_dict:
            All of the saved variables in a dictionary. This
            can be used with locals().update(data_in_dict)
            in order to load all of the data into main namespace.
    """
    import h5py

    if isinstance(path, tuple):
        exp_name = path[0]
        stimnr = str(path[1])
        exp_dir = exp_dir_fixer(exp_name)
        stim_name = getstimname(exp_dir, stimnr)
        path = os.path.join(exp_dir, 'data_analysis', stim_name,
                            stimnr+'_data.h5')

    data_in_dict = {}
    f = h5py.File(path, mode='r')
    # Get all the variable names that were saved
    keys = list(f.keys())
    for key in keys:
        item = f[key]
        # To load numpy arrays [:] trick is needed
        try:
            item = item[:]
        except ValueError:
            # This is required to load scalar values into workspace,
            # otherwise they appear as HDF5 dataset objects which are
            # not visible in variable explorer.
            try:
                item = np.asscalar(np.array(item))
            except AttributeError:
                print('%s may not be loaded properly into namespace' % key)
        data_in_dict[key] = item

    list_items = list_of_lists

    for list_item in list_items:
        if list_item in keys:
            data_in_dict[list_item] = list(data_in_dict[list_item])
    f.close()
    return data_in_dict


def load(exp_name, stimnr, fname=None):
    """
    Load data from .npz file to a dictionary, while fixing
    lists, scalars and strings.

    If the filename is something other than <stimnr>_data.npz,
    it can be supplied via fname parameter.

    locals().update(data) will load all of the variables in to
    current workspace. This does not work within functions.

    """
    stimnr = str(stimnr)
    exp_dir = exp_dir_fixer(exp_name)
    stim_name = getstimname(exp_dir, stimnr)
    if not fname:
        fname = stimnr+'_data.npz'
    path = os.path.join(exp_dir, 'data_analysis', stim_name,
                        fname)

    data_in_dict = {}
    with np.load(path) as f:
        # Get all the variable names that were saved
        keys = list(f.keys())
        for key in keys:
            item = f[key]
            if item.shape == ():
                item = np.asscalar(item)
            data_in_dict[key] = item
    # Some variables (e.g. STAs) are stored as lists originally
    # but saving to and loading from npz file converts them to
    # numpy arrays with one additional dimension. To revert
    # this, we need to turn them back into lists with list()
    # function. The variables that should be converted are
    # to be kept in list_items.
    list_items = list_of_lists

    for list_item in list_items:
        if list_item in keys:
            data_in_dict[list_item] = list(data_in_dict[list_item])
    return data_in_dict


def read_spikesheet(exp_name, cutoff=4, defaultpath=True):
    """
    Read metadata and cluster information from spike sorting file
    (manually created during spike sorting), return good clusters.

    Parameters:
    -----------
    exp_name:
        Experiment name for the directory that contains the
        .xlsx or .ods file. Possible file names may be set in
        the configuration file. Fallback/default name is
        'spike_sorting.[ods|xlsx]'.
    cutoff:
        Worst rating that is tolerated for analysis. Default
        is 4. The source of this value is manual rating of each
        cluster.
    defaultpath:
        Whether to iterate over all possible file names in exp_dir.
        If False, the full path to the file should be supplied
        in exp_name.

    Returns:
    --------
    clusters:
        Channel number, cluster number and rating of those
        clusters that match the cutoff criteria in a numpy array.
    metadata:
        Information about the experiment in a dictionary.

    Raises:
    -------
    FileNotFoundError:
        If no spike sorting file can be located.
    ValueError:
        If the spike sorting file containes incomplete information.

    Notes:
    ------
    The script assumes adherence to defined cell locations for
    metadata and cluster information. If changed undefined behavior
    may occur.
    """
    if defaultpath:
        exp_dir = exp_dir_fixer(exp_name)
        filenames = config('spike_sorting_filenames')
        for filename in filenames:
            filepath = os.path.join(exp_dir, filename)
            if os.path.isfile(filepath + '.ods'):
                filepath += '.ods'
                meta_keys = [0, 0, 1, 25]
                meta_vals = [1, 0, 2, 25]
                cluster_chnl = [4, 0, 2000, 1]
                cluster_cltr = [4, 4, 2000, 5]
                cluster_rtng = [4, 5, 2000, 6]
                break
            elif os.path.isfile(filepath + '.xlsx'):
                filepath += '.xlsx'
                meta_keys = [4, 1, 25, 2]
                meta_vals = [4, 5, 25, 6]
                cluster_chnl = [51, 1, 2000, 2]
                cluster_cltr = [51, 5, 2000, 6]
                cluster_rtng = [51, 6, 2000, 7]
                break
        else:
            raise FileNotFoundError('Spike sorting file (ods/xlsx) not found.')
    else:
        filepath = exp_name

    sheet = np.array(pyexcel.get_array(file_name=filepath, sheets=[0]))

    meta_keys = sheet[meta_keys[0]:meta_keys[2], meta_keys[1]:meta_keys[3]]
    meta_vals = sheet[meta_vals[0]:meta_vals[2], meta_vals[1]:meta_vals[3]]
    metadata = dict(zip(meta_keys.ravel(), meta_vals.ravel()))

    # Concatenate cluster information
    clusters = sheet[cluster_chnl[0]:cluster_chnl[2],
                     cluster_chnl[1]:cluster_chnl[3]]
    cl = np.argmin(clusters.shape)
    clusters = np.append(clusters,
                         sheet[cluster_cltr[0]:cluster_cltr[2],
                               cluster_cltr[1]:cluster_cltr[3]],
                         axis=cl)
    clusters = np.append(clusters,
                         sheet[cluster_rtng[0]:cluster_rtng[2],
                               cluster_rtng[1]:cluster_rtng[3]],
                         axis=cl)
    if cl != 1:
        clusters = clusters.T
    clusters = clusters[np.any(clusters != [['', '', '']], axis=1)]

    # The channels with multiple clusters have an empty line after the first
    # line. Fill the empty lines using the first line of each channel.
    for i, c in enumerate(clusters[:, 0]):
        if c != '':
            nr = c
        else:
            clusters[i, 0] = nr

    if '' in clusters:
        rowcol = (np.where(clusters == '')[1-cl][0]+1 + cluster_chnl[1-cl])
        raise ValueError('Spike sorting file is missing information in '
                         '{} {}.'.format(['column', 'row'][cl], rowcol))
    clusters = clusters.astype(int)

    # Sort the clusters in ascending order based on channel number
    # Normal sort function messes up the other columns for some reason
    # so we explicitly use lexsort for the columns containing channel nrs
    # Order of the columns given in lexsort are in reverse
    sorted_idx = np.lexsort((clusters[:, 1], clusters[:, 0]))
    clusters = clusters[sorted_idx, :]

    # Filter according to quality cutoff
    clusters = clusters[clusters[:, 2] <= cutoff]

    return clusters, metadata


def read_raster(exp_name, stimnr, channel, cluster, defaultpath=True):
    """
    Return the spike times from the specified raster file.

    Use defaultpath=False if the raster directory is not
    exp_dir + '/results/rasters/'. In this case pass the full
    path to the raster with exp_dir.
    """
    exp_dir = exp_dir_fixer(exp_name)
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
    exp_dir = exp_dir_fixer(exp_name)

    if defaultpath:
        stimdir = os.path.join(exp_dir, 'stimuli')
    else:
        stimdir = exp_dir

    # Filter stimulus directory contents with RE to allow leading zero
    pattern = f'0?{stimulusnr}_.*'
    paramfile = list(filter(re.compile(pattern).match, os.listdir(stimdir)))
    if len(paramfile) == 1:
        paramfile = paramfile[0]
    elif len(paramfile) == 0:
        raise IOError('No parameter file that starts with {} exists under'
                      ' the directory: {}'.format(stimulusnr, stimdir))
    else:
        print(paramfile)

        raise ValueError('Multiple files were found starting'
                         ' with {}'.format(stimulusnr))

    f = open(os.path.join(stimdir, paramfile))
    lines = [line.strip('\n') for line in f]
    f.close()

    parameters = {}

    parameters['filename'] = paramfile
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


def stimulisorter(exp_name):
    """
    Read parameters.txt file and return the stimuli type and
    stimuli numbers in a dictionary.
    """
    possible_stim_names = ['spontaneous', 'onoffsteps', 'fff', 'stripeflicker',
                           'checkerflicker', 'directiongratingsequence',
                           'rotatingstripes', 'frozennoise',
                           'checkerflickerplusmovie', 'OMSpatches', 'OMB',
                           'saccadegrating']
    sorted_stimuli = {key: [] for key in possible_stim_names}
    exp_dir = exp_dir_fixer(exp_name)

    file = open(os.path.join(exp_dir, 'parameters.txt'), 'r')

    for line in file:
        for stimname in possible_stim_names:
            if line.find(stimname) > 0:
                stimnr = int(line.split('_')[0])
                toadd = sorted_stimuli[stimname]
                toadd = toadd.append(stimnr)
    return sorted_stimuli


def getstimname(exp_name, stim_nr):
    """
    Returns the stimulus name for a given experiment and stimulus
    number from parameters.txt file.
    """
    exp_dir = exp_dir_fixer(exp_name)
    with open(os.path.join(exp_dir, 'parameters.txt')) as f:
        lines = f.readlines()
    lines = lines[1:]
    name = None
    for line in lines:
        if line.startswith('%s_' % stim_nr):
            name = line[:-4].strip()
        # In case the stimulus name is in the format 01_spontaneous
        elif line.startswith('0%s' % stim_nr):
            name = line[1:-4].strip()
    if name:
        return name
    else:
        raise IndexError('Stimulus {} does not exist in experiment:'
                         '\n{}'.format(stim_nr, exp_dir))


def readmat(matfile):
    """
    Read a given .mat file and return the contents in a dictionary.

    .mat files that are v>7.3 and v<7.3 must be treated differently.
    """
    data = {}
    try:
        f = scipy.io.loadmat(matfile)
        useh5 = False
    except NotImplementedError:
        useh5 = True

    if not useh5:
        for key, item in f.items():
            if not str(key).startswith('__'):
                if str(item.dtype).startswith('<U'):
                    item = str(item)
                else:
                    item = np.squeeze(item)
                data.update({key: item})
        return data
    else:
        import h5py
        with h5py.File(matfile, mode='r') as f:
            for key in f.keys():
                data.update({key: np.squeeze(f[key])})
        return data


def read_binaryfile(filepath):
    """
    Reads and returns the contents of a binary file.
    """
    with open(filepath, 'rb') as file:
        file_content = file.read()
    return file_content


def parse_binary(file_content):
    """
    Parses the binary file returned by read_binaryfile by separating
    into length and voltage trace parts.
    """
    # The header contains the length of the recording as a unsigned 32 bit int
    length = struct.unpack('I', file_content[:4])[0]
    # The rest of the binary data is the voltage trace, as unsigned 16 bit int
    voltage_raw = np.array(struct.unpack('H'*length, file_content[16:]))

    return length, voltage_raw


def readjsonfile(filename, required=False, comments=False):
    """
    Parse JSON file and strip off of annotative comments if allowed.

    Parameters
    ----------
    filename : str
        Path to a JSON file to parse
    required : bool, optional
        If False, return empty dict if file does not exist or is empty
    comments : bool, optional
        Allow annotative comments. Default is False

    Returns
    -------
    data : dict
        Parsed JSON file

    Raises
    ------
    AttributeError
        If parsing of JSON file failed due to a syntax error
    """
    if not required:
        if not os.path.isfile(filename) or os.stat(filename).st_size <= 0:
            return {}

    with open(filename, 'r') as cfile:
        data = cfile.read()

    # Remove annotative comments in default JSON (don't hate)
    if comments:
        data = re.sub(r"//.*$", "", data, flags=re.M)

    try:
        data = json.loads(data)
    except json.JSONDecodeError as je:
        apath = os.path.realpath(filename)
        raise AttributeError('Invalid syntax in the JSON file '
                             f'\'{apath}\':\n{str(je)}') from None
    return data


def nestedupdate(d, u):
    """
    Nested update for dictionaries.

    Parameters
    ----------
    d : dict
        Source dictionary to update
    u : dict
        Update dictionary with altered values
    """
    for k, v in u.items():
        if isinstance(v, collections.Mapping):
            d[k] = nestedupdate(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def cache_config():
    """
    Decorator for caching the loaded config files.
    """
    def _init(obj):
        root_path = Path(sys.modules[__name__].__file__).parents[0]
        dflt_path = root_path.joinpath('pymer_config_default.json')
        defaults = readjsonfile(dflt_path, required=True, comments=True)
        user = readjsonfile(os.path.expanduser('~/.pymer_config'))
        setattr(obj, 'cfg', nestedupdate(defaults, user))
        return obj
    return _init


@cache_config()
def config(key, default=None):
    """
    Retrieve value from loaded config.

    Parameters
    ----------
    key : int or str
        Key in the config dictionary
    default : int or str, optional
        Default value to return if key is not in config. Default is None

    Returns
    -------
    Value of the config for the given key.

    Notes
    -----
    See 'pymer_config_default.json' for more information.
    """
    return config.cfg.get(key, default)


def reload_config():
    """
    Force reloading the cached config from disk.

    Notes
    -----
    This is useful when changing the user config while running the
    functions from command line.
    """
    cfg = getattr(cache_config()(config), 'cfg')
    setattr(config, 'cfg', cfg)

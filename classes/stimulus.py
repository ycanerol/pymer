import numpy as np
from pathlib import Path

import analysis_scripts as asc
import iofuncs as iof
import plotfuncs as plf


class Stimulus:
    """
    Generic class for loading all data regarding a single stimulus. Classes for other stimuli are
    derived from this.

    Attribues:
        exp:
            Name of the experiment to be analyzed. Accepts short versions of the foldername.
        stimnr:
            Stimulus number in the experiment.
        maxframes:
            Maximum number of frames to load for the stimulus. Since the whole stimulus is loaded in the
            memory by the class, large stimuli that exceed the size of memory cause problems.
        clusters:
            List of the units in the experiments in the form of channel, cluster nr, rating
        nclusters:
            Number of units
        exp_dir:
            Full path of the main experiment folder
        exp_foldername:
            Actual folder name of the experiment
        stimname:
            Stimulus name
        clids:
            Cluster id's, padded strings with channel and cluster number, useful for plotting e.g. 02801
        frametimings:
            Pulses marking the stimulus presentation/change times in seconds.
        allspikes:
            Binned spike times for all clusters.

    """
    def __init__(self, exp, stimnr, maxframes=None):
        self.exp = exp
        self.stimnr = stimnr
        self.maxframes = maxframes
        self.clusters, self.metadata = asc.read_spikesheet(self.exp)
        self.nclusters = self.clusters.shape[0]
        self.exp_dir = Path(iof.exp_dir_fixer(exp))
        self.exp_foldername = self.exp_dir.stem
        self.stimname = iof.getstimname(self.exp_dir, self.stimnr)
        self.clids = plf.clusters_to_ids(self.clusters)
        self.refresh_rate = self.metadata['refresh_rate']
        self.sampling_rate = self.metadata['sampling_freq']
        self.readpars()
        self.get_frametimings()
        self._getstimtype()

        self.stim_dir = self.exp_dir / 'data_analysis' / self.stimname

    def _getstimtype(self):
        sortedstim = asc.stimulisorter(self.exp)

        stimtype = None
        for key, val in sortedstim.items():
            if self.stimnr in val:
                stimtype = key

        self.stimtype = stimtype

    def get_frametimings(self):
        try:
            filter_length, frametimings = asc.ft_nblinks(self.exp,
                                                         self.stimnr,
                                                         self.param_file.get('Nblinks', None),
                                                         self.refresh_rate)
        except ValueError as e:
            if str(e).startswith('Unexpected value for nblinks'):
                frametimings = asc.readframetimes(self.exp, self.stimnr)
                filter_length = None
        frametimings = frametimings[:self.maxframes]
        self.filter_length = filter_length
        self.frametimings = frametimings
        self.frame_duration = np.ediff1d(frametimings).mean()

    def readpars(self):
        self.param_file = asc.read_parameters(self.exp, self.stimnr)

    def clusterstats(self):
        print(f'There are {self.nclusters} clusters in this experiment.')
        for i in range(1, 5):
            count = self.clusters[self.clusters[:, 2] < i+1, :].shape[0]
            print(f'({100*count/self.nclusters:>5.1f}%) {count:>4} are rated {i}',
                  ' or better'*(i != 1), '.', sep='')

    def read_raster(self, i):
        ch, cl = self.clusters[i, :2]
        return asc.read_raster(self.exp, self.stimnr, ch, cl)

    def binnedspiketimes(self, i):
        return asc.binspikes(self.read_raster(i), self.frametimings)[:self.maxframes]

    def allspikes(self):
        if not self.maxframes:
            ntotal = self.binnedspiketimes(0).shape[0]
        else:
            ntotal = self.maxframes
        allspikes = np.zeros((self.nclusters, ntotal))
        for i in range(self.nclusters):
            allspikes[i] = self.binnedspiketimes(i)
        return allspikes

    def read_datafile(self):
        return iof.load(self.exp, self.stimnr)


class Parameters:
    """
    Dummy class to hold stimulus parameters.
    """
    def __init__(self):
        pass

    def __str__(self):
        return str(self.__dict__)

    def __repr__(self):
        return self.__str__()

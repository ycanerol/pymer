
import os
import numpy as np

import analysis_scripts as asc
import iofuncs as iof


class Stimulus:
    def __init__(self, exp, stimnr, maxframes=None):
        self.exp = exp
        self.stimnr = stimnr
        self.clusters, self.metadata = asc.read_spikesheet(self.exp)
        self.nclusters = self.clusters.shape[0]
        self.exp_dir = iof.exp_dir_fixer(exp)
        self.exp_foldername = os.path.split(self.exp_dir)[-1]
        self.stimname = iof.getstimname(exp, stimnr)
#        self.get_frametimings()
        self._getstimtype()
        self.refresh_rate = self.metadata['refresh_rate']
        self.sampling_rate = self.metadata['sampling_freq']
        self.maxframes = maxframes

        self.stim_dir = os.path.join(self.exp_dir, 'data_analysis',
                                     self.stimname)

    def _getstimtype(self):
        sortedstim = asc.stimulisorter(self.exp)

        stimtype = None
        for key, val in sortedstim.items():
            if self.stimnr in val:
                stimtype = key

        self.stimtype = stimtype

    def get_frametimings(self):
        frametimings = asc.readframetimes(self.exp, self.stimnr)[:self.maxframes]
        self.frametimings = frametimings

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
    Dummy class to hold parameters.
    """
    def __init__(self):
        pass

    def __str__(self):
        return str(self.__dict__)

    def __repr__(self):
        return self.__str__()

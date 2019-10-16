#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""

import numpy as np

import analysis_scripts as asc
import iofuncs as iof


from stimulus import Stimulus, Parameters
from driftinggratings import DriftingGratings

#
#def get(mydict, key, defaultval, dtype=int):
#    """
#    In the parameter file, multiple values for the same key will be retrieved
#    as string which does not play well with specifying the defaults as a
#    list of np.array in dict.get method.
#
#    This is a wrapper function that converts space separated strings into numpy
#    arrays.
#    """
#    fromstring = False
#    try:
#        retval = mydict[key]
#        if type(retval) not in (int, float):
#            fromstring = True
#    except KeyError:
#        retval = defaultval
#    if fromstring:
#        retval = np.fromstring(retval, dtype=dtype, sep=' ')
#    else:
#        retval = np.array(retval, dtype=dtype)
#    return retval

def expand_nr(arr, nrepeats):
    if isinstance(arr, int) or len(arr) == 1:
        return [arr] * nrepeats
    else:
        return arr


class DriftingGratingsSeq(Stimulus):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.stimtype != 'directiongratingsequence':
            pass # raise
            #raise ValueError('Stimulus is not direction gratings sequence')
        self.readpars()
        self._setdefaults()

    def _setdefaults(self):
        pars = Parameters()
        param_file = self.param_file
        pars.color = param_file.get('color', True)
        pars.period = param_file.get('period', [100, 60, 60, 150, 70, 15])
#        pars.period = get(param_file, 'period', [100, 60, 60, 150, 70, 15])
        pars.nrofds = len(pars.period)
        pars.nangles = param_file.get('nangles', [8, 8, 12, 4, 16, 24])
        pars.nangles = expand_nr(pars.nangles, pars.nrofds)
        pars.squarewave = param_file.get('squareWave', [0, 1, 1, 0, 0, 1])
        pars.squarewave = expand_nr(pars.squarewave, pars.nrofds)
        pars.ncycles = param_file.get('cycles', [5, 5, 5, 5, 5, 5])
        pars.ncycles = expand_nr(pars.ncycles, pars.nrofds)
        pars.pfrangles = param_file.get('preframeAngles', [300, 120, 120, 300, 120, 120])
        pars.pfrangles = expand_nr(pars.pfrangles, pars.nrofds)
        pars.preframestimuli = param_file.get('preframeStimuli', 180)
        pars.gratingwidthwhite = param_file.get(
                                     'gratingwidthwhite',
                                     [117.5, 300, 425, 120, 141, 112.5])
        pars.gratingwidthwhite = expand_nr(pars.gratingwidthwhite, pars.nrofds)
        pars.gratingwidthblack = param_file.get(
                                     'gratingwidthblack',
                                     [117.5, 900, 425, 120, 141, 112.5])
        pars.gratingwidthblack = expand_nr(pars.gratingwidthblack, pars.nrofds)
        pars.contrasts = param_file.get('contrasts', 1.0)
        pars.contrasts = expand_nr(pars.contrasts, pars.nrofds)
        pars.meanintensity = param_file.get('meanintensity', [0.5, 0.5, 0.5])
        pars.meanintensity = expand_nr(pars.meanintensity, 3)
        pars.duration = param_file.get('duration', [200, 240, 240, 600, 280, 180])
        pars.duration = expand_nr(pars.duration, pars.nrofds)
        pars.usemask = param_file.get('usemask', False)
        pars.centerx = param_file.get('centerx', 423)
        pars.centery = param_file.get('centery', 240)
        pars.radius = param_file.get('radius', 60)
        pars.maskcolor = param_file.get('maskcolor', [0.5, 0.5, 0.5])
        pars.coneisolating = param_file.get('coneIsolating', True)

        self.pars = pars

    def single_parameters(self, par_i):
        pars = self.pars
        spars = {}
        for key, val in pars.items():
            if key in ('period', 'nangles', 'squarewave', 'ncycles',
                       'pfrangles', 'preframeStimuli', 'gratingwidthwhite',
                       'gratingwidthblack', 'contrasts', 'duration', 'meanintensity'):
                newval = val[par_i]
            elif key == 'maskcolor':
                newval = val[int(par_i/2)]
            else:
                newval = val
            spars.update({key:newval})
        return spars

    def calc_ftlims(self, par_i):
        # Calculate which part of frametimings we will need
        spars = self.single_parameters(par_i)
        nperiod = int(spars['duration']/spars['period'])
        pulse_per_cycle = nperiod * spars['nangles']
        num_pulses = int(pulse_per_cycle * spars['ncycles'])
        return num_pulses

    def calc_all_frametime_lims(self):
        frametime_lims = [0]
        for par_i in range(self.pars.nrofds):
            frametime_lims.append(self.calc_ftlims(par_i))
#        frametime_lims = [sum(frametime_lims[:i]) for i in range(1, len(frametime_lims)+1)]
        frametime_lims.append(None)
        self.frametime_lims = frametime_lims

    def call_driftingratings(self, par_i):
        dg = DriftingGratings(self.exp, self.stimnr,
                              fromsequence=self.single_parameters(par_i))

if __name__ == '__main__':
    dgs = DriftingGratingsSeq('20180710', 10)

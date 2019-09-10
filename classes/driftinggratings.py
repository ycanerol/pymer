#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Drifting gratings class
"""

import numpy as np

import analysis_scripts as asc
import iofuncs as iof


from stimulus import Stimulus, Parameters


class DriftingGratings(Stimulus):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.stimtype not in ('directiongratings', 'driftinggratings'):
            raise ValueError('The stimulus is not drifting gratings.')

        self.readpars()
        self._setdefaults()

    def _setdefaults(self):
        pars = Parameters()
        param_file = self.param_file
        pars.period = param_file.get('period', 3)
        pars.ncycles = param_file.get('cycles', 3)
        pars.sequence = param_file.get('sequence', 0)
        pars.regeneration = param_file.get('regeneration', 100)
        pars.brightness = param_file.get('brightness', 1)
        pars.nangles = param_file.get('Nangles', 8)
        pars.gratingwidth = param_file.get('gratingwidth', None)
        pars.duration = param_file.get('duration', 5*pars.period)

        pars.angles = np.linspace(0, 2*np.pi, num=pars.nangles, endpoint=False)

        self.pars = pars

if __name__ == '__main__':
    exp, stimnr = 'Kuehn', 5
    dgs = Stimulus(exp, stimnr)
    dg = DriftingGratings(exp, stimnr)
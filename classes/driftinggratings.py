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
        self.reshape_frametimings()
        self.calculate_tuning_curves()
        self.vectorize_tuning_curves()
        self.calculate_dsi()

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

        pars.nperiod = int(pars.duration / pars.period)
        pars.pulse_per_cycle = pars.nperiod * pars.nangles
        ncycles = int(np.floor(self.frametimings.shape[0] / pars.pulse_per_cycle))

        if ncycles != pars.ncycles:
            raise ValueError('Unexpected number of cycles.')

        self.angles = np.linspace(0, 2*np.pi, num=pars.nangles, endpoint=False)
        self.angles_wrapped = self.wrap_first_element(self.angles)

        self.pars = pars

    def wrap_first_element(self, array):
        """
        Add the first element of an array to the end for closing polar plots.
        """
        return np.hstack((array, array[0]))

    def reshape_frametimings(self):
        ft = self.frametimings.copy()
        ft = ft[:(self.pars.pulse_per_cycle*self.pars.ncycles)]
        ft_shape = (self.pars.nperiod, self.pars.nangles, self.pars.ncycles)
        ft = ft.reshape(ft_shape)
        self.frametimings_rs = ft
        self.ft_shape = ft_shape

    def calculate_tuning_curves(self):
        binned_spikes = np.zeros((self.nclusters, *self.ft_shape))
        allspikes = self.allspikes()
        for i in range(self.nclusters):
            binned_spikes[i, ...] = allspikes[i, :].reshape(*self.ft_shape)
        self.tuning_curves_unwrapped = binned_spikes.mean(axis=(1, 3))
        self.tuning_curves = np.hstack((self.tuning_curves_unwrapped,
                                        self.tuning_curves_unwrapped[:, 0][:, None]))

    def plot_tuning_curve(self, i):
        plt.polar(self.angles_wrapped, self.tuning_curves[i, :])
        plt.plot([0, self.vec_t[i]], [0, self.vec_r[i]], 'k')

    def vectorize_tuning_curves(self):
        y_all = np.sin(self.angles)[None, ...] * self.tuning_curves_unwrapped
        x_all = np.cos(self.angles)[None, ...] * self.tuning_curves_unwrapped
        y = y_all.sum(axis=1)
        x = x_all.sum(axis=1)
        self.vec_r = np.hypot(x, y)
        self.vec_t = np.arctan2(y, x)  # HINT: order of y and x is important

    def calculate_dsi(self):
        dsi = self.vec_r/self.tuning_curves_unwrapped.sum(axis=1)
        self.dsi = dsi

if __name__ == '__main__':
    exp, stimnr = 'Kuehn', 5
    dgs = Stimulus(exp, stimnr)
    dg = DriftingGratings(exp, stimnr)
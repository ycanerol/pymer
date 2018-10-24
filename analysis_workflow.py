#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 15:21:30 2018

@author: ycan
"""
import datetime
import sys

from . import analyze
from . import plotcheckerstas, plotcheckersurround, plotcheckersvd
from .modules import analysisfuncs as asc


def experiment(exp_name=''):
    """
    Run all analyzers on an experiment

    Parameters
    ----------
    exp_name : str, optional
        Path to experiment or substring of experiment name
    """
    if exp_name == '':
        # Attempt to read from stdin
        try:
            exp_name = str(sys.argv[1])
        except IndexError:
            exp_name = input('Enter the experiment name to be analyzed: ')

    start_time = datetime.datetime.now().strftime('%A %X')
    print(f'Analysis started on {start_time}')
    sorted_stimuli = asc.stimulisorter(exp_name)

    spontaneous = sorted_stimuli['spontaneous']
    fullfieldflicker = sorted_stimuli['fff']
    onoffsteps = sorted_stimuli['onoffsteps']
    checkerflicker = sorted_stimuli['checkerflicker']
    stripeflicker = sorted_stimuli['stripeflicker']
    checkerflickerplus = (sorted_stimuli['frozennoise']
                          + sorted_stimuli['checkerflickerplusmovie'])
    OMSpatches = sorted_stimuli['OMSpatches']
    OMB = sorted_stimuli['OMB']
    saccadegrating = sorted_stimuli['saccadegrating']

    asc.saveframetimes(exp_name)

    # %%
    analyze.spont(exp_name, spontaneous)

    analyze.fff(exp_name, fullfieldflicker)
    analyze.allfff(exp_name, fullfieldflicker)

    analyze.onoffsteps(exp_name, onoffsteps)
    analyze.allonoff(exp_name, onoffsteps)

    analyze.stripeflicker(exp_name, stripeflicker)
    analyze.stripesurround(exp_name, stripeflicker)

    for stim in OMSpatches:
        analyze.omspatches(exp_name, stim)
    for stim in OMB:
        analyze.omb(exp_name, stim)
    for stim in saccadegrating:
        analyze.saccadegratings(exp_name, stim)

    # %%
    for i in checkerflicker:
        analyze.checkerflicker(exp_name, i)
        plotcheckerstas(exp_name, i)
        plotcheckersurround(exp_name, i)
        plotcheckersvd(exp_name, i)

    for i in checkerflickerplus:
        analyze.checkerflickerplus(exp_name, i)
        plotcheckerstas(exp_name, i)
        plotcheckersurround(exp_name, i)
        plotcheckersvd(exp_name, i)

    end_time = datetime.datetime.now().strftime('%A %X')
    print(f'Analysis completed on {end_time}')

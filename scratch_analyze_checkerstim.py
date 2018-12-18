#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 20:38:42 2018

@author: ycan
"""

import analysis_scripts as asc
from checkerflickeranalyzer import checkerflickeranalyzer
from plot_checker_stas import plot_checker_stas
from plotcheckersvd import plotcheckersvd
from plotcheckersurround import plotcheckersurround
from checkerflickerplusanalyzer import checkerflickerplusanalyzer

exp_name = '20180710'

sorted_stimuli = asc.stimulisorter(exp_name)

checkerflicker = sorted_stimuli['checkerflicker']

checkerflickerplus = sorted_stimuli['frozennoise'] + sorted_stimuli['checkerflickerplusmovie']
checkerflickerplus = sorted_stimuli['checkerflickerplusmovie']

for i in checkerflickerplus:
    checkerflickerplusanalyzer(exp_name, i)
    plot_checker_stas(exp_name, i)
    plotcheckersurround(exp_name, i)
    plotcheckersvd(exp_name, i)

#for i in checkerflicker:
#    checkerflickeranalyzer(exp_name, i)
#    plot_checker_stas(exp_name, i)
#    plotcheckersurround(exp_name, i)
#    plotcheckersvd(exp_name, i)
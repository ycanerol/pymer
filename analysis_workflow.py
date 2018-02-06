#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 15:21:30 2018

@author: ycan
"""
import sys
import analysis_scripts as asc
from spontanalyzer import spontanalyzer
from fffanalyzer import fffanalyzer
from onoffstepsanalyzer import onoffstepsanalyzer
from checkerflickeranalyzer import checkerflickeranalyzer
from plot_checker_stas import plot_checker_stas
from plotcheckersvd import plotcheckersvd
from plotcheckersurround import plotcheckersurround
from stripeflickeranalysis import stripeflickeranalysis
from plotstripestas import plotstripestas
from allonoff import allonoff
from stripesurround import stripesurround

# Attempt to read from stdin
try:
    exp_name = str(sys.argv[1])
except IndexError:
    exp_name = input('Enter the experiment name to be analyzed: ')

sorted_stimuli = asc.stimulisorter(exp_name)

spontaneous = sorted_stimuli['spontaneous']
fullfieldflicker = sorted_stimuli['fff']
onoffsteps = sorted_stimuli['onoffsteps']
checkerflicker = sorted_stimuli['checkerflicker']
stripeflicker = sorted_stimuli['stripeflicker']

asc.saveframetimes(exp_name)

# %%
spontanalyzer(exp_name, spontaneous)

fffanalyzer(exp_name, fullfieldflicker)

onoffstepsanalyzer(exp_name, onoffsteps)
allonoff(exp_name, onoffsteps)

stripeflickeranalysis(exp_name, stripeflicker)
plotstripestas(exp_name, stripeflicker)
stripesurround(exp_name, stripeflicker)

# %%
for i in checkerflicker:
    checkerflickeranalyzer(exp_name, i)
    plot_checker_stas(exp_name, i)
    plotcheckersurround(exp_name, i)
    plotcheckersvd(exp_name, i)

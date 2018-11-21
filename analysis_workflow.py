#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 15:21:30 2018

@author: ycan
"""
import sys
import datetime
import analysis_scripts as asc
from spontanalyzer import spontanalyzer
from fffanalyzer import fffanalyzer
from allfff import allfff
from onoffstepsanalyzer import onoffstepsanalyzer
from checkerflickeranalyzer import checkerflickeranalyzer
from plot_checker_stas import plot_checker_stas
from plotcheckersvd import plotcheckersvd
from plotcheckersurround import plotcheckersurround
from stripeflickeranalysis import stripeflickeranalysis
from allonoff import allonoff
from stripesurround import stripesurround
from checkerflickerplusanalyzer import checkerflickerplusanalyzer
from OMSpatchesanalyzer import OMSpatchesanalyzer
from OMBanalyzer import OMBanalyzer
from saccadegratingsanalyzer import saccadegratingsanalyzer

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
checkerflickerplus = sorted_stimuli['frozennoise'] + sorted_stimuli['checkerflickerplusmovie']
OMSpatches = sorted_stimuli['OMSpatches']
OMB = sorted_stimuli['OMB']
saccadegrating = sorted_stimuli['saccadegrating']

asc.saveframetimes(exp_name)

# %%
spontanalyzer(exp_name, spontaneous)

fffanalyzer(exp_name, fullfieldflicker)
allfff(exp_name, fullfieldflicker)

onoffstepsanalyzer(exp_name, onoffsteps)
allonoff(exp_name, onoffsteps)

stripeflickeranalysis(exp_name, stripeflicker)
stripesurround(exp_name, stripeflicker)

OMSpatchesanalyzer(exp_name, OMSpatches)
for stim in OMB:
    OMBanalyzer(exp_name, stim)
for stim in saccadegrating:
    saccadegratingsanalyzer(exp_name, stim)

# %%
for i in checkerflicker:
    checkerflickeranalyzer(exp_name, i)
    plot_checker_stas(exp_name, i)
    plotcheckersurround(exp_name, i)
    plotcheckersvd(exp_name, i)

for i in checkerflickerplus:
    checkerflickerplusanalyzer(exp_name, i)
    plot_checker_stas(exp_name, i)
    plotcheckersurround(exp_name, i)
    plotcheckersvd(exp_name, i)


end_time = datetime.datetime.now().strftime('%A %X')
print(f'Analysis completed on {end_time}')

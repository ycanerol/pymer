#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 15:21:30 2018

@author: ycan
"""
import analysis_scripts as asc

exp_name = '20180124'

spontaneous = [1, 2, 7, 13]
fullfieldflicker = [4, 10, 15]
onoffsteps = [3, 8, 14]
checkerflicker = [5, 11, 16]
stripeflicker = [6, 12, 17]

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

# %%
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

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 15:21:30 2018

@author: ycan
"""
import analysis_scripts as asc

exp_name = '2018'

spontaneous = [1, 2, 8, 15]
fullfieldflicker = [4, 12, 17]
onoffsteps = [3, 9, 10, 16]
checkerflicker = [5, 6, 13, 18]
stripeflicker = [7, 14, 19]

runfile('/home/ycan/Documents/scripts/onoffstepsanalyzer.py')
#  %%
asc.saveframetimes(exp_name)

# %%
for i in spontaneous:
    spontanalyzer(exp_name, i)

# %%
for i in fullfieldflicker:
    fffanalyzer(exp_name, i)

# %%
for i in onoffsteps:
    onoffstepsanalyzer(exp_name, i)

# %%
for i in checkerflicker:
    checkerflickeranalyzer(exp_name, i)
    plot_checker_stas(exp_name, i)
    plotcheckersvd(exp_name, i)
    plotcheckersurround(exp_name, i)

# %%
for i in stripeflicker:
    stripeflickeranalysis(exp_name, i)
    plot_stripestas(exp_name, i)
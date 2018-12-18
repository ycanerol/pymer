#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 18:53:21 2018

@author: ycan
"""
#import sys
#import time
#
#
#w=50
#itemnr = 750
#
#for i in range(itemnr+1):
##    sys.stdout.flush()
#    prog = i/itemnr
#    bar_complete = int(prog*w)
#    bar_noncomplete = w-bar_complete
#    sys.stdout.write('\r{}{} |{:4.1f}%'.format('█'*bar_complete,
#                     '-'*bar_noncomplete,
#                     prog*100))
#    time.sleep(.1)
#sys.stdout.write('\n')

#%%
import analysis_scripts as asc
from checkerflickeranalyzer import checkerflickeranalyzer
exp_name = '20180802'
sorted_stimuli = asc.stimulisorter(exp_name)
checkerflicker = sorted_stimuli['checkerflicker']
checkerkwargs = {'clusterstoanalyze':10, 'frametimingsfraction':.05}
checkerflickeranalyzer(exp_name, checkerflicker[0], **checkerkwargs)

#%%
import analysis_scripts as asc
from checkerflickerplusanalyzer import checkerflickerplusanalyzer
exp_name = '20180802'
sorted_stimuli = asc.stimulisorter(exp_name)
checkerflickerplus = sorted_stimuli['frozennoise'] + sorted_stimuli['checkerflickerplusmovie']
checkerkwargs = {'clusterstoanalyze':10, 'frametimingsfraction':.05}
checkerflickerplusanalyzer(exp_name, checkerflickerplus[0], **checkerkwargs)

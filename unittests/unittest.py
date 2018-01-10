#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 14:52:09 2018

@author: ycan
"""
#%%
import iofuncs as iof
stimname = iof.stimname

def nametests():
    stimname = iof.stimname

    assert stimname('20171116', 6) == '6_checkerflicker5x5bw1blink'
    assert stimname('20171122', 6) == '6_checkerflicker5x5bw1blink'
    assert stimname('20171122', 7) == '7_checkerflicker5x5bw2blinks'
    assert stimname('V', 10) == '10_checkerflicker_2x2bw_4blinks'


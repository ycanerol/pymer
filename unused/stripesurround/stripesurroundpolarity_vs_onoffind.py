#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 10:43:08 2018

@author: ycan

Compare the polarities (onoroff, based on absmax of sta,
used to determine whether to flip for fitting and plotting) from
stripesurround with onoffindex from onoffsteps stimulus.

It is expected that most should agree but it is not the case actually.
No further action is taken so far about it.
"""
import iofuncs as iof
import matplotlib.pyplot as plt

exp_name = '20180124'
pairs = [(3, 6),(8, 12)]
conditions = ['Low', 'High']

for i, pair in enumerate(pairs):
    onoffbias = iof.load(exp_name, pair[0])['onoffbias']
    data = iof.load(exp_name, pair[1])
    quals = data['quals']
    cs_inds = data['cs_inds']
    polarities = data['polarities']
    plt.scatter(onoffbias, polarities, label=conditions[i])
    plt.xlabel('On-Off bias')
    plt.ylabel('Polarity index')

#    plt.ylabel('Center-Surround index')
    plt.legend()

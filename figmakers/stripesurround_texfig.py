#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 12:08:14 2018

@author: ycan
"""


import texplot
import iofuncs as iof
import analysis_scripts as asc
import miscfuncs as msc
import plotfuncs as plf

import matplotlib.pyplot as plt
import numpy as np

from stripesurround import onedgauss

fig = texplot.texfig(1.2)

exp_name = '20180207'
stimnr = 12

exp_dir = iof.exp_dir_fixer(exp_name)

_, metadata = asc.read_ods(exp_dir)
px_size = metadata['pixel_size(um)']

data = iof.load(exp_name, stimnr)

clusters = data['clusters']
stas = data['stas']
max_inds = data['max_inds']
filter_length = data['filter_length']
stx_w = data['stx_w']
exp_name = data['exp_name']
stimname = data['stimname']
frame_duration = data['frame_duration']
quals = data['quals']
all_parameters = data['all_parameters']
fits = data['fits']

index = 5
sta = data['stas'][index]
max_i = data['max_inds'][index]
onoroff = data['polarities'][index]
fit = fits[index]
popt = all_parameters[index]

cut_time = int(100/(frame_duration*1000)/2)
fsize = int(700/(stx_w*px_size))
t = np.arange(filter_length)*frame_duration*1000
vscale = fsize * stx_w*px_size

sta, max_i = msc.cutstripe(sta, max_i, fsize*2)


ax1 = fig.add_subplot(121)
plf.subplottext('A', ax1)
plf.stashow(sta, ax1, extent=[0, t[-1], -vscale, vscale])
ax1.set_xlabel('Time [ms]')
ax1.set_ylabel(r'Distance [$\upmu$m]')

fitv = np.mean(sta[:, max_i[1]-cut_time:max_i[1]+cut_time+1],
               axis=1)

s = np.arange(fitv.shape[0])

ax2 = fig.add_subplot(122)
plf.subplottext('B', ax2, x=-.1)
plf.spineless(ax2)
ax2.set_yticks([])
ax2.set_xticks([])
ax2.plot()
ax2.plot(onoroff*fitv, -s, label='Data')
ax2.plot(onedgauss(s, *popt[:3]), -s,  '--', label='Center')
ax2.plot(-onedgauss(s, *popt[3:]), -s,  '--', label='Surround')


plt.savefig('/home/ycan/Downloads/sta.pdf', bbox_inches='tight')
plt.show()

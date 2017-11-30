#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 11:52:37 2017

@author: ycan
"""
import plotfuncs as plf
raster = np.resize(spikes, (800,))

ax =plt.subplot(211)
plt.eventplot(spikes, orientation='horizontal')
plf.spineless(ax)
ax = plt.subplot(212)
plt.eventplot(spikes)
plf.spineless(ax)


a = np.array([[1,2,3],[3, 4],[5, 6,3,2,6,3,6,2,5,7]])
a[0]

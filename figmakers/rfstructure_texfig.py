#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 18:17:27 2018

@author: ycan
"""

import numpy as np
import matplotlib.pyplot as plt
import plotfuncs as plf
import texplot


r = 35
x = np.linspace(-r, r, num=200)
def f(x, a, mu, sig):
    y = a*np.exp((-(x-mu)**2)/(2*sig**2))
#    y = a*np.exp(-np.power((x-mu), 2)/(2 * sig**2))
    return y

y1 = f(x, .3, 0, 3)
y2 = f(x, -.1, 0, 9)

fig = texplot.figsize(.9)
plf.addarrowaxis(plt.gca(), x=0.1, y=0.4, dx=0.1, dy=0.2,
                 xtext='Distance', ytext='Sensitivity',
                 xtextoffset=.02, ytextoffset=.032, fontsize='x-small')
plt.plot(x, y1+y2)
plt.plot(x, y1, '--')
plt.plot(x, y2, '--')
plt.xticks([])
plt.yticks([])
texplot.savefig('rfstructure')
plt.show()

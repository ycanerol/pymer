#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 17:24:20 2017

@author: ycan

Try to define a custom colormap that has black in the middle and white towards
extremes with color in the middle
"""
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('/home/ycan/Documents/scripts/modules/')
import plotfuncs as plf


red = np.concatenate(([0]*16, np.linspace(1, 0, 15)))

green = np.concatenate((np.linspace(1, 0, 16)[:-1], np.linspace(0, 1, 16)))
green = [0]*31

blue = np.concatenate((np.linspace(1, 0, 16), [0]*15))
blue = red[::-1]

colors = []
for i in range(31):
    colors.append([red[i], green[i], blue[i]])

cm = plf.RFcolormap(colors)

im = plt.imshow(np.random.randint(0, 100, (100, 100)), cmap=cm)
plt.colorbar(im)

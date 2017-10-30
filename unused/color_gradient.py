#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 15:37:49 2017

@author: ycan
"""
import numpy as np
import matplotlib.pyplot as plt

x, y = np.meshgrid(np.linspace(0, 80, 80), np.linspace(0, 60, 60))
d = np.sqrt(x*x+y*y)
sigma, mu = 20.0, 20.0
g = np.exp(-( (d-mu)**2 / ( 40.0 * sigma**2 ) ) )

plt.imshow(g)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 16:46:57 2017

@author: ycan

Draw rings corresponding to different sigma distances on the receptive field

https://stackoverflow.com/questions/20892251/contours-with-map-overlay-on
-irregular-grid-in-python
"""
import numpy as np
from matplotlib.mlab import griddata
import matplotlib.pyplot as plt

x_lin = np.linspace(fit_frame.min(), fit_frame.max(), f_size)
y_lin = np.linspace(fit_frame.min(), fit_frame.max(), f_size)
x_lin, y_lin = np.meshgrid(x_lin, y_lin)

z_grid =

# Convert Z values from the Gaussian distribution into standard deviations
# a.k.a. Mahalonobis distance
Zm = np.log((Z-pars[0])/pars[1])
# Round the SD values for
Zmr

plt.imshow(fit_frame)
plt.contour(Zmf, levels=np.arange(-5,0))
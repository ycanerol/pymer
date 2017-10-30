#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 16:32:53 2017

@author: ycan

Used for fitting a 2D gaussian for getting the surround
as done in Cowan et al 2016

Copied from http://scipy-cookbook.readthedocs.io/items/FittingData.html
"""
import numpy as np
from scipy import optimize


def gaussian(height, center_x, center_y, width_x, width_y):
    """Returns a gaussian function with the given parameters"""
    width_x = float(width_x)
    width_y = float(width_y)
    return lambda x, y: height*np.exp(
                -(((center_x-x)/width_x)**2+((center_y-y)/width_y)**2)/2)


def moments(data):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution by calculating its
    moments """
    total = data.sum()
    X, Y = np.indices(data.shape)
    x = (X*data).sum()/total
    y = (Y*data).sum()/total
    col = data[:, int(y)]
    width_x = np.sqrt(np.abs((np.arange(col.size)-y)**2*col).sum()/col.sum())
    row = data[int(x), :]
    width_y = np.sqrt(np.abs((np.arange(row.size)-x)**2*row).sum()/row.sum())
    height = data.max()
    return height, x, y, width_x, width_y


def fitgaussian(data):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution found by a fit"""
    params = moments(data)
    errorfunction = lambda p: np.ravel(gaussian(*p)(*np.indices(data.shape)) -
                                       data)
    p, success = optimize.leastsq(errorfunction, params)
    return p

f = 30

#fit_frame = fit_frame2[max_i[0]-f:max_i[0]+f, max_i[1]-f:max_i[1]+f]
#
#fit_frame=fit_frame2

#plt.imshow(fit_frame)
p = fitgaussian(fit_frame)
fit_func = gaussian(*p)

X, Y = np.meshgrid(np.arange(fit_frame.shape[0]), np.arange(fit_frame.shape[1]))

Z = fit_func(Y,X)

plt.imshow(Z)
#plt.contour(fit_func(*np.indices(fit_frame.shape)), cmap=plt.cm.copper)
plt.show()
plt.imshow(fit_frame)
plt.show()
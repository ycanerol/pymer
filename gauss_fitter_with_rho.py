#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 12:09:00 2017

@author: ycan

Taken from
http://scipy-cookbook.readthedocs.io/items/FittingData.html
and modified.

The other alternative, a more advanced version of the same package gaussfitter
by Adam Ginsburg, uses rotation parameter for fitting. This is not ideal as
we require the covariance matrix of the distribution (not the parameters that
is returned in mpfit object in gaussfit package) to be able to calculate
distances in terms of sigma (Mahalonobis distance). Therefore now rho
(correlation coefficient) will be added to the simpler version.

Abandoned: rho is not needed for Mahalonibis distance. One can use the exponent
of the fitted function as the distance.


"""
import numpy as np
from scipy import optimize


def gaussian(height, center_x, center_y, width_x, width_y, rho):
    """Returns a gaussian function with the given parameters"""
    width_x = float(width_x)
    width_y = float(width_y)
#    return lambda x, y: height*np.exp(
#                -(((center_x-x)/width_x)**2+((center_y-y)/width_y)**2)/2)
    return lambda x, y: height*((2*np.pi*width_x*width_y*np.sqrt(1-rho**2))**-1*
                        np.exp((2*(1-rho**2)**-1)*((((x-center_x)/width_x)**2)+
                                (((y-center_y)/width_y)**2)-
                                (2*rho*(y-center_y)*(x-center_x)/width_x*width_y))/-2))

#    return lambda x, y: height*((2*np.pi*width_x*width_y*np.sqrt(1-rho**2))**-1)*np.exp(((-2*(1-rho**2))**-1)*((((x-center_x)/width_x)**2)+(((y-center_y)/width_y)**2)-((2*rho*(x-center_x)*(y-center_y))/(width_x*width_y))))


def moments(data):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution by calculating its
    moments """
    total = np.abs(data).sum()
    Y, X = np.indices(data.shape)  # python convention: reverse x,y np.indices
    y = np.argmax((X*np.abs(data)).sum(axis=1)/total)
    x = np.argmax((Y*np.abs(data)).sum(axis=0)/total)
    col = data[:, int(y)]
    width_x = np.sqrt(np.abs((np.arange(col.size)-y)**2*col).sum()/col.sum())
    row = data[int(x), :]
    width_y = np.sqrt(np.abs((np.arange(row.size)-x)**2*row).sum()/row.sum())
    height = data.max()
    rho = 0.001
    return height, x, y, width_x, width_y, rho


def fitgaussian(data):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution found by a fit"""
    params = moments(data)
    errorfunction = lambda p: np.ravel(gaussian(*p)(*np.indices(data.shape)) -
                                       data)
    p, success = optimize.leastsq(errorfunction, params)
    return p

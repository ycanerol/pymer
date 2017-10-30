#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 13:29:08 2017

@author: ycan
"""
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

stride=4

plotthis = Zm

#ax.contour(X, Y, plotthis, stride=stride)
ax.plot_surface(X, Y, plotthis, alpha=.3, color='red')

cset = ax.contour(X, Y, plotthis, zdir='z', offset=-40, cmap=cm.gnuplot, stride=stride)
#cset = ax.contour(X, Y, plotthis, zdir='x', offset=0, cmap=cm.gnuplot)
#cset = ax.contour(X, Y, plotthis, zdir='y', offset=f_size*2, cmap=cm.gnuplot)


plt.show()

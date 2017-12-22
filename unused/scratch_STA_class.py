#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 16:50:49 2017

@author: ycan
"""
import numpy as np
import matplotlib.pyplot as plt


class STA:
    def __init__(self, data, scale, dt):
        self.data = data
        self.maxi = np.squeeze(np.where(np.abs(data) == np.abs(np.max(data))))
        # Size of one pixel of STA in micrometers
        self.scale = scale
        # Time between each frame
        self.dt = dt

    def show(self, cm='RdBu', onlymax=True):
        a = self.data
        vmax = np.max(np.abs([np.min(a), np.max(a)]))
        vmin = -vmax
        inds = self.maxi[2]
        for i in range(inds):
            plt.imshow(a[:, :, i], vmax=vmax, vmin=vmin, cmap=cm)
        plt.show()

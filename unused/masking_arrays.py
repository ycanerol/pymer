#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 13:32:46 2017

@author: ycan
"""

import numpy as np
import matplotlib.pyplot as plt

size_x = 60
size_y = 80
size_t = 20

center = [30, 40, 2]

a = np.random.randint(-19, 19, (size_x, size_y, size_t))
#
#mask = np.array([True]*size_x*size_y*size_t)
#
#mask = mask.reshape(size_x, size_y, size_t)
#
#mask[3:6, 3:6, :] = False
#
#mask[4:5, 4:5, :] = True
#
##ama = np.ma.masked_array(a, mask)
#
#asd = np.ma.array(a, mask=mask)
#
#
#plt.subplot(1,2,1); plt.imshow(a[:,:,0])
#plt.subplot(1,2,2); plt.imshow(asd[:,:,0])
#plt.show();

# Copied to get_surround.py
#def ringmask(data, center_px, r):
#    mask = np.array([True]*data.size)
#    mask = np.reshape(mask, data.shape)
#
#    cx = center_px[0]
#    cy = center_px[1]
#
#    mask[cx-r:cx+r, cy-r:cy+r, :] = False
#    mask[cx-(r-1):cx+(r-1), cy-(r-1):cy+(r-1), :] = True
#
#    masked_data = np.ma.array(data, mask=mask)
#
#    return masked_data

plt.plot()
for i in range(10):
    plt.imshow(ringmask(a, center ,i)[0][:, :, 0])
    plt.title(i)
    plt.show()

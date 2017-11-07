#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 10:36:16 2017

@author: ycan
"""
import numpy as np
import matplotlib.pyplot as plt

slims = np.arange(0, 8, step=.2)

for i, item in enumerate(slims, 1):
    # Define upper and lower bounds
    upper = item
    lower = slims[i-2]
    mask = np.logical_not(np.logical_and(Zm >= lower, Zm <= upper))
    mask3d = np.broadcast_arrays(sta, mask[..., None])[1]
    plt.plot(np.mean(np.ma.array(sta, mask=mask3d), axis=(0, 1)) - i/20,
             label=str(upper))
    plt.axhline(-i/20, alpha=.4, color='k', linestyle='dashed')
    plt.text(-3, -i/20, '{:2.2f}-{:2.2f}'.format(lower, upper), size=8)
#plt.legend()
plt.axis('off')
plt.show()

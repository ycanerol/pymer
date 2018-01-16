#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 15:05:25 2017

@author: ycan
"""
import matplotlib.pyplot as plt

temporal = []

for i in range(len(stas)):
    a = stas[i]
    b = max_inds[i]

    temporal.append(a[b[0], b[1], :])

for i, filt in enumerate(temporal):
    plt.subplot(11, 9, i+1)
    plt.plot(np.arange(filter_length)*frame_duration*1000, filt)
    plt.xlabel('Time[ms]')
    plt.axis('off')
plt.show()
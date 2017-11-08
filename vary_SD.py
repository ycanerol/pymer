#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 10:36:16 2017

@author: ycan
"""
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('ggplot')

slims = np.arange(0, 10, step=.6)
filters = np.zeros((len(slims),20))

for i in range(1, len(slims)):
    # Define upper and lower bounds
    upper = slims[i]
    lower = slims[i-1]
    mask = np.logical_not(np.logical_and(Zm >= lower, Zm <= upper))
    mask3d = np.broadcast_arrays(sta, mask[..., None])[1]
    component = np.mean(np.ma.array(sta, mask=mask3d), axis=(0, 1))
    filters[i-1] = component
    offset = -i/20
    plt.plot(component + offset,
             label=str(upper))
    plt.axhline(offset, alpha=.4, color='k', linestyle='dashed')
    plt.text(-3, offset, '{:2.2f}-{:2.2f}'.format(lower, upper), size=8)
#plt.legend()
plt.axis('off')
plt.show()

# %%

peaks = np.argmax(np.abs(filters), axis=1)
mexican = np.array([])
for peak, i in enumerate(peaks):
    mexican = np.append(mexican, filters[peak, i])
mexican = np.append(mexican[::-1], mexican[1:])
slims2 = np.append(-slims[::-1], slims[1:])
ax = plt.subplot(111)
ax.plot(slims2, mexican)
plt.xlabel('Distance from RF center[$\sigma$]')
plt.ylabel('Relative sensitivity')
plt.title('Mexican hat?')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)
plt.show()

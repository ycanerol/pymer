#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 11:07:28 2017

@author: ycan

Weigh the STA using inverse of Mahalonobis distance to prevent surround from
being averaged out.
"""

weights = np.ma.array(1/Zm, mask=surround_mask)
weights = weights/np.max(weights)
#weights = weights/np.ma.sqrt(np.ma.sum(np.ma.power(weights, 2)))

#Inelegant but functional solution
weights3d = weights
for i in range(sta.shape[2]-1):
    weights3d = np.ma.dstack((weights3d, weights))

weighted_surround = np.mean(sta*weights3d, axis=(0, 1))

plt.plot(weighted_surround, label='weighted surround')
plt.plot(sta_center_temporal, label='center')
plt.plot(sta_surround_temporal, label='surround')
plt.legend()
plt.show()
plt.imshow(weights3d[:,:,1])
plt.show()
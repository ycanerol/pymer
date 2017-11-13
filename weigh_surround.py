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

#Normalized according to their abs max values
sta_ctn = sta_center_temporal/np.max(np.abs(sta_center_temporal))
sta_stn = sta_surround_temporal/np.max(np.abs(sta_surround_temporal))
sta_wsn = weighted_surround/np.max(np.abs(weighted_surround))

plt.plot(sta_wsn, label='weighted surround')
plt.plot(sta_ctn, label='center')
plt.plot(sta_stn, label='surround')
plt.legend()
plt.show()
plt.imshow(weights3d[:,:,1])
plt.show()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 16:50:49 2017

@author: ycan
"""
import numpy as np
import matplotlib.pyplot as plt
import plotfuncs as plf



class STA:
    def __init__(self, sta): #, scale, dt):
        self.sta = sta
        self.maxi = np.squeeze(np.where(np.abs(sta) == np.abs(np.max(sta))))
        self.quality = self._calcquality()
        # Size of one pixel of STA in micrometers
#        self.scale = scale
        # Time between each frame
#        self.dt = dt

    def show(self, cm='RdBu', onlymax=True):
        a = self.sta
        vmax = a[tuple(self.maxi)]
        vmin = -vmax
        inds = self.sta.shape[-1]
        rows, columns = plf.numsubplots(inds)
        for i in range(inds):
            ax = plt.subplot(rows, columns, i+1)
            ax.imshow(a[:, :, i], vmax=vmax, vmin=vmin, cmap=cm)
        plt.show()

    def _calcquality(self):
        """
        z-score of the brightest pixel
        """
        a = self.sta
        z = (np.max(np.abs(a)) - a.mean()) / a.std()
        return z.astype('float16')

#%%
import iofuncs as iof

exps = [('V', 10), ('Kara', 5), ('20171116', 6), ('20171122', 6), ('20171122', 7)]
#all_quals = np.empty(len(exps))
all_quals = []

for i in range(len(exps)):
    exp = exps[i]
    data = iof.loadh5(exp)

    stas = data['stas']

    stasc = []

    for sta in stas:
        stasc.append(STA(sta))

    qual = []
    for sta in stasc:
        qual = np.append(qual, sta.quality)
    all_quals.append(qual)
#all_quals = all_quals[1:, :]
#%%
for j in range(len(all_quals)):
    plt.scatter(all_quals[j], [j]*len(all_quals[j]))
    plt.text(50, j, str(exps[j]), fontsize=8)
plt.show()

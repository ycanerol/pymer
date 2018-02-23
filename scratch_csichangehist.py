#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 15:38:19 2018

@author: ycan
"""
import plotfuncs as plf

toplot = ~np.isnan(csi[0, :])
csih = csi[:, toplot]
colorsh = [color for i, color in enumerate(colors) if toplot[i]]

onmask = np.array([True if color=='blue' else False for color in colorsh])
offmask = np.array([True if color=='red' else False for color in colorsh])

csion = csih[:, onmask]
csioff = csih[:, offmask]

fig = plt.figure(figsize=(12,12))
plt.title(r'Center-Surround Index Change ($csi_{photopic} - csi_{mesopic}$)')
axes = fig.subplots(3, 1, sharex=True)
for ax, cs, color in zip(axes, [csih, csion, csioff], [None, 'blue', 'red']):
    ax.hist(cs[1, :]-cs[0, :], bins=30, color=color, alpha=.8)
#    plf.spineless(ax)
#plt.hist(csih[1, :]-csih[0, :], bins=30)
#plt.show()
#plt.hist(csion[1, :]-csion[0, :], bins=30, color='blue')
#plt.show()
#plt.hist(csioff[1, :]-csioff[0, :], bins=30, color='red')
plt.show()


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
unstable_mask = np.array([True if color=='orange' else False for color in colorsh])

csion = csih[:, onmask]
csioff = csih[:, offmask]
csi_unstable = csih[:, unstable_mask]

#fig = plt.figure(figsize=(12,12))
fig = texplot.texfig(.9)


axes = fig.subplots(4, 1, sharex=True)
plt.suptitle('Center-Surround Index Change ($csi_{photopic} - csi_{mesopic}$)')
for ax, cs, color in zip(axes, [csih, csion, csioff, csi_unstable],
                         [None, 'blue', 'red', 'orange']):
    ax.hist(cs[1, :]-cs[0, :], bins=np.linspace(-.8, .8, 30),
            color=color, alpha=.8)
    plf.spineless(ax)
    ax.set_ylim([0, 20])
    ax.set_yticks(np.linspace(0, 20, 3))

#plt.hist(csih[1, :]-csih[0, :], bins=30)
#plt.show()
#plt.hist(csion[1, :]-csion[0, :], bins=30, color='blue')
#plt.show()
#plt.hist(csioff[1, :]-csioff[0, :], bins=30, color='red')
plt.savefig('/home/ycan/Downloads/asd.pdf')
plt.show()

